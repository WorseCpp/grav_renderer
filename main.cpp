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

constexpr int SCR_WIDTH = 800;
constexpr int SCR_HEIGHT = 600;
constexpr float sphereRadius = 10000.0f;
constexpr float M = .0f;

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
                // glm::vec2(-0.25f, -0.25f),
                // glm::vec2( 0.25f, -0.25f),
                // glm::vec2(-0.25f,  0.25f),
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
                rays.emplace_back(position, ray_direction, glm::vec3(.0f), 0.0f);
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

float du_by_dphi_squared(float r, float u)
{
    return 1.5 * (2 * M) * u * u - u;
}

// Computes one RK45 step for the second-order ODE:
//   d²r/dphi² = f(r, phi)
// by rewriting it as a system:
//   dr/dphi = u,  du/dphi = f(r, phi)
// r0   - current value of r
// dr0  - current value of dr/dphi
// phi0 - current phi value
// h    - step size in phi
// f    - function providing the second derivative given r and phi
// Returns a tuple: { new_r, new_dr/dphi, new_phi }
std::tuple<float, float, float> rk45_step(float r0, float dr0, float phi0, float h, 
    const std::function<float(float, float)>& f) {

    // k1 for both r and dr/dphi.
    float k1_r  = h * dr0;
    float k1_dr = h * f(r0, phi0);

    // k2 computed at phi0 + h/4.
    float k2_r  = h * (dr0 + 0.25 * k1_dr);
    float k2_dr = h * f(r0 + 0.25 * k1_r, phi0 + 0.25 * h);

    // k3 computed at phi0 + 3h/8.
    float k3_r  = h * (dr0 + (3.0/32.0) * k1_dr + (9.0/32.0) * k2_dr);
    float k3_dr = h * f(r0 + (3.0/32.0) * k1_r + (9.0/32.0) * k2_r, phi0 + (3.0/8.0) * h);

    // k4 computed at phi0 + (12/13)*h.
    float k4_r  = h * (dr0 + (1932.0/2197.0) * k1_dr - (7200.0/2197.0) * k2_dr + (7296.0/2197.0) * k3_dr);
    float k4_dr = h * f(r0 + (1932.0/2197.0) * k1_r - (7200.0/2197.0) * k2_r + (7296.0/2197.0) * k3_r, phi0 + (12.0/13.0) * h);

    // k5 computed at phi0 + h.
    float k5_r  = h * (dr0 + (439.0/216.0)*k1_dr - 8.0 * k2_dr + (3680.0/513.0)*k3_dr - (845.0/4104.0)*k4_dr);
    float k5_dr = h * f(r0 + (439.0/216.0)*k1_r - 8.0 * k2_r + (3680.0/513.0)*k3_r - (845.0/4104.0)*k4_r, phi0 + h);

    // k6 computed at phi0 + h/2.
    float k6_r  = h * (dr0 - (8.0/27.0)*k1_dr + 2.0 * k2_dr - (3544.0/2565.0)*k3_dr + (1859.0/4104.0)*k4_dr - (11.0/40.0)*k5_dr);
    float k6_dr = h * f(r0 - (8.0/27.0)*k1_r + 2.0 * k2_r - (3544.0/2565.0)*k3_r + (1859.0/4104.0)*k4_r - (11.0/40.0)*k5_r, phi0 + 0.5 * h);

    // Fourth-order estimate.
    float r4  = r0  + (25.0/216.0)*k1_r + (1408.0/2565.0)*k3_r + (2197.0/4104.0)*k4_r - (1.0/5.0)*k5_r;
    float dr4 = dr0 + (25.0/216.0)*k1_dr + (1408.0/2565.0)*k3_dr + (2197.0/4104.0)*k4_dr - (1.0/5.0)*k5_dr;

    // Fifth-order estimate.
    float r5  = r0  + (16.0/135.0)*k1_r + (6656.0/12825.0)*k3_r + (28561.0/56430.0)*k4_r - (9.0/50.0)*k5_r + (2.0/55.0)*k6_r;
    float dr5 = dr0 + (16.0/135.0)*k1_dr + (6656.0/12825.0)*k3_dr + (28561.0/56430.0)*k4_dr - (9.0/50.0)*k5_dr + (2.0/55.0)*k6_dr;

    float new_phi = phi0 + h;
    // Return the fifth-order estimates as the new state.
    return { r5, dr5, new_phi };
}

glm::vec2 u_phi_to_x_y(float u, float phi)
{
    float r = 1 / u;
    float x = r * std::cos(phi);
    float y = r * std::sin(phi);
    return glm::vec2(x, y);
}

Ray linearize_ray(const Ray& ray) {

    double r0 = glm::length(ray.position);

    double x0 = r0;

    double y0 = .0;

    double phi0 = .0;

    double u0 = 1. / r0;

    glm::vec3 new_pos = ray.position + ray.direction * 0.01f; // Set the initial position in the xy plane.

    double r1 = glm::length(new_pos);

    double phi1 = glm::acos(glm::dot(new_pos, ray.position) / (r0 * r1));
    
    double u1 = 1.0 / r1;

    double du_dphi = (u1 - u0) / (phi1 - phi0); // Approximate the radial derivative.

    while (true) { // Iterate until the derivative is small enough.
        
        auto [new_u, new_du_dphi, new_phi] = rk45_step(u0, du_dphi, phi0, 0.001, du_by_dphi_squared);
        
        u0 = new_u;
        
        du_dphi = new_du_dphi;
        
        phi0 = new_phi;

        if (std::abs(u0) > .9 * sphereRadius)
            break;

        if (u0 != u0){
            return Ray(glm::vec3(sphereRadius * 10), glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0), 0);
        }

        if (1 / u0 < 2 * M)
        {
            return Ray(glm::vec3(sphereRadius * 10), glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0), 0);
        }
    }

    float phi = phi0;

    float r = 1 / u0;

    float x_off = r * glm::cos(phi);
    
    float y_off = r * glm::sin(phi);
    
    glm::vec3 new_ray_pos = ray.position + x_off * glm::normalize(ray.position) + y_off * glm::normalize(ray.direction);

    glm::vec2 new_ray_0 = u_phi_to_x_y(u0, phi0);
    
    glm::vec2 new_ray_1 = u_phi_to_x_y(u0 + du_dphi * 0.01, phi0 + 0.01);
    
    glm::vec2 new_dir = new_ray_1 - new_ray_0;

    glm::vec3 new_ray_dir = new_dir.x * glm::normalize(ray.position) + new_dir.y * glm::normalize(ray.direction);
    
    return Ray(new_ray_pos, new_ray_dir, ray.color, ray.time);
}

glm::vec3 sample(const std::vector<Ray>& rays, const std::vector<Star>& stars) {
    glm::vec3 accumulatedColor(0.0f);
    
    for (const auto& rayIm : rays) {
        bool hit = false;

        auto ray = linearize_ray(rayIm); // Linearize the ray to approximate its path.

        std::cout << "Linearized ray: position(" << ray.position.x << ", " << ray.position.y << ", " << ray.position.z 
              << "), direction(" << ray.direction.x << ", " << ray.direction.y << ", " << ray.direction.z << ")\n";
        
        std::cout << "Original ray: position(" << rayIm.position.x << ", " << rayIm.position.y << ", " << rayIm.position.z 
              << "), direction(" << rayIm.direction.x << ", " << rayIm.direction.y << ", " << rayIm.direction.z << ")\n";

        // Check each star for a collision (ray-sphere intersection)
        for (const auto& star : stars) {
            // Compute the vector from the ray's origin to the star's center.
            if (linear_approx_collides(ray, star)) {
                // If the ray collides with the star, accumulate its color.
                accumulatedColor += star.c * star.l; // Assuming star.l is the light intensity.
                break; // Stop checking other stars once we hit one.
            }
        }
    }
    
    // Return the average color from all sample rays.
    return accumulatedColor / static_cast<float>(rays.size());
}



static std::mt19937 gen(std::random_device{}());
int main() {


    Ray testRay(glm::vec3(100.0f, 0.0f, 0.0f), glm::normalize(glm::vec3(-.9f, 0.1f, 0.0f)), glm::vec3(1.0f), 0.0f);
    std::cout << "Created ray at (" 
              << testRay.position.x << ", " << testRay.position.y << ", " << testRay.position.z << ")"
              << " with direction (" 
              << testRay.direction.x << ", " << testRay.direction.y << ", " << testRay.direction.z << ")" 
              << std::endl;

              
    auto linray = linearize_ray(testRay);

    std::cout << "Linearized ray at (" 
              << linray.position.x << ", " << linray.position.y << ", " << linray.position.z << ")"
              << " with direction (" 
              << linray.direction.x << ", " << linray.direction.y << ", " << linray.direction.z << ")"
              << " and color (" 
              << linray.color.r << ", " << linray.color.g << ", " << linray.color.b << ")" 
              << std::endl;

    glm::vec3 pos(-10.0f, 0.0f, 5.0f);
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

    //for (int j = 0; j < 10; j++)
    //{
    //    // Choose non-zero initial phi to avoid division by zero.
    //    double r0 = 1.0;
    //    double phi0 = 0.0;
    //    double dr0 = 1.0;
    //    double h = 0.01;
//
    //    // Define f from the second derivative:
    //    // Differentiating dr/dphi = sqrt(phi) gives: d²r/dphi² = 1/(2*sqrt(phi))
    //    auto f = [](double r, double phi) {
    //        return phi;
    //    };
//
    //    auto step = rk45_step(r0, dr0, phi0, h, f);
//
    //    std::cout << "\nRK45 step result:" << std::endl;
    //    std::cout << "r = " << std::get<0>(step) << std::endl;
    //    std::cout << "dr/dphi = " << std::get<1>(step) << std::endl;
    //    std::cout << "phi = " << std::get<2>(step) << std::endl;
    //    std::cout << std::exp(std::get<2>(step)) << std::endl;
    //}

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

    std::cout << "Generated " << stars.size() << " stars." << std::endl;
    for (size_t i = 0; i < stars.size(); ++i) {
        const Star& star = stars[i];
        std::cout << "Star " << i << ": Position(" 
                  << star.pos.x << ", " << star.pos.y << ", " << star.pos.z << "), "
                  << "Color(" << star.c.r << ", " << star.c.g << ", " << star.c.b << "), "
                  << "Radius(" << star.r << ")" << std::endl;
    }

    //stars = {Star{1.0f, glm::vec3(1000.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), 100.0f}};


    // Example usage of Camera within main
    {

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
        dumpPhoto("output.ppm", camera.capturedScreen);
    }

    std::cout << "\n";

    return 0;
}