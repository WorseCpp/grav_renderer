#include <iostream>

#include <fstream>

#include <glm/glm.hpp>

#include <vector>

#include <string>

#include <random>

#include <functional>

#include <cmath>
#include <thread>
#include <chrono>

#include <utility>

constexpr int scr = 5;

int SCR_WIDTH = 800 / scr;
int SCR_HEIGHT = 600 / scr;
float sphereRadius = 1000.0f;

float p = 30;

float M = std::pow(10, -p);

struct Star {
    float l;
    glm::vec3 pos;
    glm::vec3 c;
    float r;
};

struct Ray {
    public:
        glm::vec3 position;
        glm::vec3 direction;
        glm::vec3 color;
        float time;

        Ray(const glm::vec3 &position, const glm::vec3 &direction, const glm::vec3 &color, float time)
            : position(position), direction(direction), color(color), time(time) {}

        // Default constructor
        Ray() : position(0.0f), direction(0.0f), color(0.0f), time(0.0f) {}
};

void dumpPhoto(const std::string& filename, const std::vector<std::vector<glm::vec3>>& image) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }
    int height = image.size();
    int width = height > 0 ? image[0].size() : 0;
    
    // Write header for PPM (P3 is a plain text format)
    ofs << "P3\n" << width << " " << height << "\n255\n";

    // Write pixel data (each component scaled to [0, 255])
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int r = static_cast<int>(255.99f * image[y][x].r);
            int g = static_cast<int>(255.99f * image[y][x].g);
            int b = static_cast<int>(255.99f * image[y][x].b);
            ofs << r << " " << g << " " << b << "  ";
        }
        ofs << "\n";
    }
    ofs.close();
}

class Camera {
    public:
        glm::vec3 position;
        glm::vec3 front;
        glm::vec3 right;
        float ratio;
        float aspectRatio;
        float fov; // Field of view in degrees.
        std::vector<std::vector<glm::vec3>> capturedScreen; // 2D array representing the captured screen.

        // Constructs a Camera using a given up vector for calculating the right vector.
        // The ratio, aspectRatio, fov, and screen dimensions default to preset values if not provided.
        Camera(const glm::vec3& pos, const glm::vec3& frontVec, const glm::vec3& up, 
               float ratio = 1.0f, float aspectRatio = 1.0f, float fov = 45.0f, 
               size_t screenWidth = SCR_WIDTH, size_t screenHeight = SCR_HEIGHT)
            : position(pos), front(glm::normalize(frontVec)), ratio(ratio), aspectRatio(aspectRatio), fov(fov) {
            right = glm::normalize(glm::cross(front, up));
            capturedScreen.resize(screenHeight, std::vector<glm::vec3>(screenWidth, glm::vec3(0.0f)));
        }

        // Default constructor with typical default values.
        Camera()
            : position(0.0f, 0.0f, 0.0f),
              front(0.0f, 0.0f, -1.0f),
              right(1.0f, 0.0f, 0.0f),
              ratio(1.0f),
              aspectRatio(1.0f),
              fov(45.0f) {
            // Default screen dimensions: SCR_WIDTH x SCR_HEIGHT.
            capturedScreen.resize(SCR_HEIGHT, std::vector<glm::vec3>(SCR_WIDTH, glm::vec3(0.0f)));
        }

        std::vector<Ray> get_rays(int i, int j)
        {

            std::vector<Ray> rays;

            size_t screenHeight = capturedScreen.size();
            size_t screenWidth = capturedScreen[0].size();

            // Compute the up vector from the camera orientation.
            glm::vec3 up = glm::normalize(glm::cross(right, front));

            // Convert field of view to radians and compute the scale.
            float fov_rad = glm::radians(fov);
            float scale = tan(fov_rad * 0.5f);

            // For antialiasing, generate 4 sample rays within the pixel.
            // Note: i is the row (vertical index) and j is the column.
            std::vector<glm::vec2> offsets = {
                glm::vec2(-0.25f, -0.25f),
                glm::vec2( 0.25f, -0.25f),
                glm::vec2(-0.25f,  0.25f),
                glm::vec2( 0.25f,  0.25f)
            };

            for (const auto& offset : offsets) {
                // Calculate normalized device coordinates (NDC) for the sample.
                float u = (static_cast<float>(j) + 0.5f + offset.x) / static_cast<float>(screenWidth);
                float v = (static_cast<float>(i) + 0.5f + offset.y) / static_cast<float>(screenHeight);

                // Transform NDC to screen space coordinates.
                // x coordinate scaled by aspect ratio and fov, y coordinate flipped.
                float x_ndc = (2.0f * u - 1.0f) * aspectRatio * scale;
                float y_ndc = (1.0f - 2.0f * v) * scale;

                // Compute the ray direction from the camera through the pixel sample.
                glm::vec3 ray_direction = glm::normalize(front + x_ndc * right + y_ndc * up);

                // Create the ray starting at the camera position.
                rays.emplace_back(position, ray_direction, glm::vec3(.0f), 1.0f);
            }

            return rays;
        }
};


bool linear_approx_collides(const Ray& r, const Star& s) {
    // Compute the vector from the ray's origin to the star's center.
    glm::vec3 oc = glm::normalize(s.pos - r.position);
    
    float dir = glm::dot(oc, r.direction);
    
    float theta = glm::acos(dir);

    float max_theta = atan2(s.r, glm::length(s.pos - r.position));
    
    return (theta < max_theta && theta > 0.f);
}

float du_by_dphi_squared(float u, float phi)
{
    return 1.5 * (2 * M) * u * u - u;
}

std::tuple<float, float, float> RK4(float u, float phi, float du_by_dphi, float h)
{
    // Our state is Y = {u, v} with v = du_by_dphi.
    // The ODE system is:
    // u' = v
    // v' = du_by_dphi_squared(u, phi)
    auto f = [&](float u_val, float v_val, float phi_val) -> std::pair<float, float>
    {
        return { v_val, du_by_dphi_squared(u_val, phi_val) };
    };

    // Compute k1 = h * f(u, v, phi)
    float k1_u, k1_v;
    {
        auto [du, dv] = f(u, du_by_dphi, phi);
        k1_u = h * du;
        k1_v = h * dv;
    }

    // k2 at phi + h/4 with state = {u + k1_u/4, v + k1_v/4}
    float k2_u, k2_v;
    {
        float u2 = u + k1_u * (1.0f/4.0f);
        float v2 = du_by_dphi + k1_v * (1.0f/4.0f);
        auto [du, dv] = f(u2, v2, phi + h/4.0f);
        k2_u = h * du;
        k2_v = h * dv;
    }

    // k3 at phi + 3h/8 with state = {u + 3/32*k1_u + 9/32*k2_u, v + 3/32*k1_v + 9/32*k2_v}
    float k3_u, k3_v;
    {
        float u3 = u + (3.0f/32.0f)*k1_u + (9.0f/32.0f)*k2_u;
        float v3 = du_by_dphi + (3.0f/32.0f)*k1_v + (9.0f/32.0f)*k2_v;
        auto [du, dv] = f(u3, v3, phi + 3.0f*h/8.0f);
        k3_u = h * du;
        k3_v = h * dv;
    }

    // k4 at phi + 12h/13 with state = {u + 1932/2197*k1_u - 7200/2197*k2_u + 7296/2197*k3_u, ...}
    float k4_u, k4_v;
    {
        float u4 = u + (1932.0f/2197.0f)*k1_u - (7200.0f/2197.0f)*k2_u + (7296.0f/2197.0f)*k3_u;
        float v4 = du_by_dphi + (1932.0f/2197.0f)*k1_v - (7200.0f/2197.0f)*k2_v + (7296.0f/2197.0f)*k3_v;
        auto [du, dv] = f(u4, v4, phi + 12.0f*h/13.0f);
        k4_u = h * du;
        k4_v = h * dv;
    }

    // k5 at phi + h with state = {u + 439/216*k1_u - 8*k2_u + 3680/513*k3_u - 845/4104*k4_u, ...}
    float k5_u, k5_v;
    {
        float u5 = u + (439.0f/216.0f)*k1_u - 8.0f*k2_u + (3680.0f/513.0f)*k3_u - (845.0f/4104.0f)*k4_u;
        float v5 = du_by_dphi + (439.0f/216.0f)*k1_v - 8.0f*k2_v + (3680.0f/513.0f)*k3_v - (845.0f/4104.0f)*k4_v;
        auto [du, dv] = f(u5, v5, phi + h);
        k5_u = h * du;
        k5_v = h * dv;
    }

    // k6 at phi + h/2 with state = {u - 8/27*k1_u + 2*k2_u - 3544/2565*k3_u + 1859/4104*k4_u - 11/40*k5_u, ...}
    float k6_u, k6_v;
    {
        float u6 = u - (8.0f/27.0f)*k1_u + 2.0f*k2_u - (3544.0f/2565.0f)*k3_u + (1859.0f/4104.0f)*k4_u - (11.0f/40.0f)*k5_u;
        float v6 = du_by_dphi - (8.0f/27.0f)*k1_v + 2.0f*k2_v - (3544.0f/2565.0f)*k3_v + (1859.0f/4104.0f)*k4_v - (11.0f/40.0f)*k5_v;
        auto [du, dv] = f(u6, v6, phi + h/2.0f);
        k6_u = h * du;
        k6_v = h * dv;
    }

    // Combine to produce a 5th order solution:
    float u_new = u + (16.0f/135.0f)*k1_u + (6656.0f/12825.0f)*k3_u + (28561.0f/56430.0f)*k4_u
                  - (9.0f/50.0f)*k5_u + (2.0f/55.0f)*k6_u;

    float du_by_dphi_new = du_by_dphi + (16.0f/135.0f)*k1_v + (6656.0f/12825.0f)*k3_v + (28561.0f/56430.0f)*k4_v
                           - (9.0f/50.0f)*k5_v + (2.0f/55.0f)*k6_v;

    return std::tuple<float, float, float>(u_new, phi + h, du_by_dphi_new);
}

std::pair<float, float> u_phi_to_x_y(float u, float phi)
{
    float r = 1 / u;
    float x = r * std::cos(phi);
    float y = r * std::sin(phi);
    return std::pair<float, float>(x, y);
}

std::pair<float, float> x_y_to_u_phi(float x, float y)
{
    float r = sqrt(x*x + y*y);
    float u = 1. / r;
    float phi = atan2(y, x);
    return std::pair<float, float>(u, phi);
}

Ray linearize_ray(const Ray& ray) {

    double r0 = glm::length(ray.position);

    float x0 = r0;

    float y0 = .0;

    double phi0, u0;
    
    glm::vec3 i_hat = glm::normalize(ray.position);

    glm::vec3 plane_normal = glm::normalize(glm::cross(i_hat, ray.direction));
    
    glm::vec3 j_hat = glm::cross(plane_normal, i_hat);

    assert(std::abs(glm::dot(i_hat, ray.position) - glm::length(ray.position)) < .01);

    assert(std::abs(glm::dot(i_hat, j_hat)) < .01);
    
    std::tie(u0, phi0) = x_y_to_u_phi(x0, y0);

    glm::vec3 new_pos = ray.position + ray.direction * 0.0001f; // Set the initial position in the xy plane.

    float x1 = glm::dot(new_pos, i_hat);
    float y1 = glm::dot(new_pos, j_hat);

    float u1, phi1;

    std::tie(u1, phi1) = x_y_to_u_phi(x1, y1);

    float du_dphi = (u1 - u0) / (phi1 - phi0);

    assert(glm::length(x0 * i_hat + y0 * j_hat - ray.position) < .01);
    
    assert(glm::length(x1 * i_hat + y1 * j_hat - new_pos) < .01);

    assert(glm::dot(j_hat, ray.direction) > 0);
    
    //std::cout << x0 <<", " << y0 << "," << x1 << " ," << y1 << "\n";

    int ctr = 0;

    while (true) {
        
        float h = .1;
        redo:
        int strike = 0;
        auto [new_u, new_phi, new_du_dphi] = RK4(u0, phi0, du_dphi, h);
        strike ++;

        if (new_u < 0)
        {
            h /= 5;

            if (strike > 5)
            {
                return Ray(glm::vec3(sphereRadius * 10), glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0), 0);
            }

            goto redo;
        }


        //std::cout << "["<<new_u << ", " << new_phi << "],\n";

        if (std::abs(1 / new_u) > .9 * sphereRadius)
        {

            float a = ((.9 * sphereRadius) - u0) / (new_u - u0);

            assert(abs((1-a)*u0 + a * new_u - .9*sphereRadius) < .01 * sphereRadius);

            u0 = .9 * sphereRadius;

            du_dphi = (1-a) * du_dphi + a * new_du_dphi;

            phi0 = (1-a) * phi0 + a * new_phi;

            break;
        }   

        u0 = new_u;
        
        du_dphi = new_du_dphi;
        
        phi0 = new_phi;


        if (u0 != u0){
            //std::cout << " Nana Out\n";
            return Ray(glm::vec3(sphereRadius * 10), glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0), 0);
        }

        if (std::abs(1 / u0) < 2 * M)
        {
            //std::cout << "fell in!\n";
            return Ray(glm::vec3(sphereRadius * 10), glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0), 0);
        }
    }

    float xf, yf;
    std::tie(xf, yf) = u_phi_to_x_y(u0, phi0);

    float uf1 = u0 + du_dphi * .0001;
    float phif1 = phi0 + .0001;

    float xf1, yf1;
    std::tie(xf1, yf1) = u_phi_to_x_y(uf1, phif1);

    //std::cout << xf <<", " << yf << ", " << xf1 << ", " << yf1 << "\n";

    glm::vec3 new_ray_pos = i_hat * xf1 + j_hat * yf1;

    glm::vec3 new_ray_dir = glm::normalize(i_hat * (xf1 - xf) + j_hat * (yf1 - yf));
    
    return Ray(new_ray_pos, new_ray_dir, ray.color, ray.time);
}

glm::vec3 sample(const std::vector<Ray>& rays, const std::vector<Star>& stars) {
    glm::vec3 accumulatedColor(0.0f);
    
    for (const auto& rayIm : rays) {
        bool hit = false;

        auto ray = linearize_ray(rayIm); // Linearize the ray to approximate its path.

        //std::cout << "Linearized ray: position(" << ray.position.x << ", " << ray.position.y << ", " << ray.position.z 
        //      << "), direction(" << ray.direction.x << ", " << ray.direction.y << ", " << ray.direction.z << ")\n";
        //
        //std::cout << "Original ray: position(" << rayIm.position.x << ", " << rayIm.position.y << ", " << rayIm.position.z 
        //      << "), direction(" << rayIm.direction.x << ", " << rayIm.direction.y << ", " << rayIm.direction.z << ")\n";

        // Check each star for a collision (ray-sphere intersection)
        for (const auto& star : stars) {
            // Compute the vector from the ray's origin to the star's center.
            if (linear_approx_collides(ray, star)) {
                // If the ray collides with the star, accumulate its color.
                accumulatedColor += star.c * star.l; // Assuming star.l is the light intensity.
                break; // Stop checking other stars once we hit one.
            }
            else if (glm::length(ray.position) < sphereRadius)
            {
                accumulatedColor += glm::vec3(.003f);
            }
        }
    }
    
    // Return the average color from all sample rays.
    return accumulatedColor / static_cast<float>(rays.size());
}



static std::mt19937 gen(1); //std::random_device{}());
int main() {


    Ray testRay(glm::vec3(100.0f, 0.0f, 0.0f), glm::normalize(glm::vec3(-.9f, 0.1f, 0.0f)), glm::vec3(1.0f), 0.0f);
    //std::cout << "Created ray at (" 
    //          << testRay.position.x << ", " << testRay.position.y << ", " << testRay.position.z << ")"
    //          << " with direction (" 
    //          << testRay.direction.x << ", " << testRay.direction.y << ", " << testRay.direction.z << ")" 
    //          << std::endl;

              
    auto linray = linearize_ray(testRay);

    //std::cout << "Linearized ray at (" 
    //          << linray.position.x << ", " << linray.position.y << ", " << linray.position.z << ")"
    //          << " with direction (" 
    //          << linray.direction.x << ", " << linray.direction.y << ", " << linray.direction.z << ")"
    //          << " and color (" 
    //          << linray.color.r << ", " << linray.color.g << ", " << linray.color.b << ")" 
    //          << std::endl;

    glm::vec3 pos(-10.0f, 0.0f, 0.0f);
    glm::vec3 front(1.0f, 0.0f, 0.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    
    // Initialize camera with explicit screen dimensions and aspect ratio.
    Camera camera(pos, front, up, 1.0f, static_cast<float>(SCR_WIDTH) / static_cast<float>(SCR_HEIGHT), 90.0f, SCR_WIDTH, SCR_HEIGHT);


    std::vector<Star> stars; // Populate with stars if desired.
            
    // Populate stars evenly distributed on a sphere of radius 10000
    const int numStars = 100;
    const float goldenRatio = (1.0f + std::sqrt(5.0f)) / 2.0f;
    const float angleIncrement = 2.0f * 3.14159265358979323846f * goldenRatio;


    std::uniform_real_distribution<float> offsetDis(-0.1f, 0.1f);
    std::uniform_real_distribution<float> dis(.1f, 1.f);

 
    for (int i = 0; i < numStars; i++) {
        float t = static_cast<float>(i) / static_cast<float>(numStars);
        float inclination = std::acos(1.0f - 2.0f * t);
        float azimuth = angleIncrement * i;
        {

            inclination += offsetDis(gen);
            azimuth += offsetDis(gen);
        }
        
        glm::vec3 pos;
        pos.x = sphereRadius * std::sin(inclination) * std::cos(azimuth);
        pos.y = sphereRadius * std::sin(inclination) * std::sin(azimuth);
        pos.z = sphereRadius * std::cos(inclination);
        
        Star star;
        star.l = 1.0f;
        star.pos = pos;
        star.c = glm::vec3(1.0f);  // White color

        star.r = glm::exp(-dis(gen)) * 30.f;
        
        {
            glm::vec3 toStar = glm::normalize(star.pos - camera.position);
            float halfFovRadians = glm::radians(camera.fov * 0.5f);
            if (glm::dot(camera.front, toStar) < std::cos(halfFovRadians)) {
                continue;
            }
        }

        stars.push_back(star);
    }


    // stars = {Star{1.0f, glm::vec3(1000.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), 100.0f}};


    std::cout << "Generated " << stars.size() << " stars." << std::endl;
    for (size_t i = 0; i < stars.size(); ++i) {
        const Star& star = stars[i];
        std::cout << "Star " << i << ": Position(" 
                  << star.pos.x << ", " << star.pos.y << ", " << star.pos.z << "), "
                  << "Color(" << star.c.r << ", " << star.c.g << ", " << star.c.b << "), "
                  << "Radius(" << star.r << ")" << std::endl;
    }


    // Example usage of Camera within main
    for (p; p > 0; p--){
        M = std::pow(10, -p);

        // Fill capturedScreen with a simple gradient image.
        for (size_t y = 0; y < camera.capturedScreen.size(); y++) {
            std::cout << "\rProgress: " << (100.0 * y / camera.capturedScreen.size()) << "%" << std::flush;
            for (size_t x = 0; x < camera.capturedScreen[0].size(); x++) {
            
                auto rays = camera.get_rays(y, x);
                glm::vec3 color = sample(rays, stars);
                camera.capturedScreen[y][x] = color;
            }
        }

        // Dump the generated photo to 'output.ppm'
        std::stringstream ss;
        ss << "gr-e" << p <<"-" << scr<< ".ppm";
        dumpPhoto(ss.str(), camera.capturedScreen);
    }

    std::cout << "\n";

    return 0;
}