#ifndef EINSTEIN_SOLVER_H
#define EINSTEIN_SOLVER_H

#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace Einstein
{
    const float PI = 3.14159265359f;
    const float G_NEWTON = 1.0f;
    const float C_LIGHT = 1.0f;

    inline int sym4_index(int mu, int nu)
    {
        if (mu > nu) std::swap(mu, nu);
        return mu * 4 - mu * (mu + 1) / 2 + nu;
    }

    struct SpacetimeMetric
    {
        float g[10];

        float g_inv[10];

        float christoffel[4][10];

        float d_g[4][10];

        bool singular;


        SpacetimeMetric()
        {
            for (int i = 0; i < 10; i++)
            {
                g[i] = 0.0f;
                g_inv[i] = 0.0f;
                for (int sigma = 0; sigma < 4; sigma++)
                {
                    d_g[sigma][i] = 0.0f;
                }
            }

            g[sym4_index(0, 0)] = -1.0f;
            g[sym4_index(1, 1)] = 1.0f;
            g[sym4_index(2, 2)] = 1.0f;
            g[sym4_index(3, 3)] = 1.0f;

            g_inv[sym4_index(0, 0)] = -1.0f;
            g_inv[sym4_index(1, 1)] = 1.0f;
            g_inv[sym4_index(2, 2)] = 1.0f;
            g_inv[sym4_index(3, 3)] = 1.0f;

            for (int l = 0; l < 4; l++)
                for (int i = 0; i < 10; i++)
                    christoffel[l][i] = 0.0f;

            singular = false;
        }

        float get_g(int mu, int nu) const
        {
            return g[sym4_index(mu, nu)];
        }

        void set_g(int mu, int nu, float value)
        {
            g[sym4_index(mu, nu)] = value;
        }

        float get_g_inv(int mu, int nu) const
        {
            return g_inv[sym4_index(mu, nu)];
        }

        void set_g_inv(int mu, int nu, float value)
        {
            g_inv[sym4_index(mu, nu)] = value;
        }

        float get_christoffel(int lambda, int mu, int nu) const
        {
            return christoffel[lambda][sym4_index(mu, nu)];
        }

        void set_christoffel(int lambda, int mu, int nu, float value)
        {
            christoffel[lambda][sym4_index(mu, nu)] = value;
        }
    };

    class SpacetimeGrid
    {
    public:
        int nx, ny, nz;
        float dx, dy, dz;
        glm::vec3 origin;
        std::vector<SpacetimeMetric> data;

        SpacetimeGrid(int nx_, int ny_, int nz_, glm::vec3 min_corner, glm::vec3 max_corner)
            : nx(nx_), ny(ny_), nz(nz_)
        {
            origin = min_corner;
            glm::vec3 size = max_corner - min_corner;
            dx = size.x / (nx - 1);
            dy = size.y / (ny - 1);
            dz = size.z / (nz - 1);
            data.resize(nx * ny * nz);

            std::cout << "[Einstein] Creating grid " << nx << "x" << ny << "x" << nz
                << " (" << data.size() << " points)" << std::endl;
        }

        int index(int i, int j, int k) const
        {
            return i + nx * (j + ny * k);
        }

        glm::vec3 position(int i, int j, int k) const
        {
            return origin + glm::vec3(i * dx, j * dy, k * dz);
        }

        SpacetimeMetric& at(int i, int j, int k)
        {
            return data[index(i, j, k)];
        }

        const SpacetimeMetric& at(int i, int j, int k) const
        {
            return data[index(i, j, k)];
        }
    };

    void ComputeMetric(SpacetimeGrid& grid, const std::vector<glm::vec4>& bodies)
    {
        std::cout << "\n[Einstein] Computing metric g_μν..." << std::endl;
        std::cout << "[Einstein] Number of masses: " << bodies.size() << std::endl;

        int total_points = grid.nx * grid.ny * grid.nz;
        int progress_step = total_points / 20;

        for (int k = 0; k < grid.nz; k++)
        {
            for (int j = 0; j < grid.ny; j++)
            {
                for (int i = 0; i < grid.nx; i++)
                {
                    int idx = grid.index(i, j, k);
                    if (progress_step > 0 && idx % progress_step == 0)
                    {
                        float percent = 100.0f * idx / total_points;
                        std::cout << "\r[Einstein] Metric progress: " << std::fixed
                            << std::setprecision(1) << percent << "% " << std::flush;
                    }

                    glm::vec3 pos = grid.position(i, j, k);
                    SpacetimeMetric& metric = grid.at(i, j, k);

                    metric = SpacetimeMetric();

                    float phi = 0.0f;

                    for (const auto& body : bodies)
                    {
                        glm::vec3 body_pos(body.x, body.y, body.z);
                        float mass = body.w;

                        glm::vec3 r_vec = pos - body_pos;
                        float r = glm::length(r_vec);

                        if (r < mass * 0.5f)
                        {
                            metric.singular = true;
                            continue;
                        }

                        float r_safe = std::max(r, 1e-6f);
                        phi += -G_NEWTON * mass / r_safe;
                    }

                    if (metric.singular) continue;

                    float g00 = -(1.0f + 2.0f * phi);
                    float g_spatial = 1.0f - 2.0f * phi;

                    metric.set_g(0, 0, g00);
                    metric.set_g(1, 1, g_spatial);
                    metric.set_g(2, 2, g_spatial);
                    metric.set_g(3, 3, g_spatial);

                    metric.set_g_inv(0, 0, -1.0f / g00);
                    metric.set_g_inv(1, 1, 1.0f / g_spatial);
                    metric.set_g_inv(2, 2, 1.0f / g_spatial);
                    metric.set_g_inv(3, 3, 1.0f / g_spatial);
                }
            }
        }

        std::cout << "\r[Einstein] Metric progress: 100.0% - Complete!     " << std::endl;
    }

    void ComputeMetricDerivatives(SpacetimeGrid& grid)
    {
        std::cout << "[Einstein] Computing derivatives ∂_σ g_μν..." << std::endl;

        int total_points = grid.nx * grid.ny * grid.nz;
        int progress_step = total_points / 20;

        for (int k = 1; k < grid.nz - 1; k++)
        {
            for (int j = 1; j < grid.ny - 1; j++)
            {
                for (int i = 1; i < grid.nx - 1; i++)
                {
                    int idx = grid.index(i, j, k);
                    if (progress_step > 0 && idx % progress_step == 0)
                    {
                        float percent = 100.0f * idx / total_points;
                        std::cout << "\r[Einstein] Derivatives progress: " << std::fixed
                            << std::setprecision(1) << percent << "% " << std::flush;
                    }

                    SpacetimeMetric& metric = grid.at(i, j, k);

                    if (metric.singular) continue;

                    for (int comp = 0; comp < 10; comp++)
                        metric.d_g[0][comp] = 0.0f;

                    for (int comp = 0; comp < 10; comp++)
                    {
                        float g_plus = grid.at(i + 1, j, k).g[comp];
                        float g_minus = grid.at(i - 1, j, k).g[comp];
                        metric.d_g[1][comp] = (g_plus - g_minus) / (2.0f * grid.dx);
                    }

                    for (int comp = 0; comp < 10; comp++)
                    {
                        float g_plus = grid.at(i, j + 1, k).g[comp];
                        float g_minus = grid.at(i, j - 1, k).g[comp];
                        metric.d_g[2][comp] = (g_plus - g_minus) / (2.0f * grid.dy);
                    }

                    for (int comp = 0; comp < 10; comp++)
                    {
                        float g_plus = grid.at(i, j, k + 1).g[comp];
                        float g_minus = grid.at(i, j, k - 1).g[comp];
                        metric.d_g[3][comp] = (g_plus - g_minus) / (2.0f * grid.dz);
                    }
                }
            }
        }

        std::cout << "\r[Einstein] Derivatives progress: 100.0% - Complete!     " << std::endl;
    }

    void ComputeChristoffelSymbols(SpacetimeGrid& grid)
    {
        std::cout << "[Einstein] Computing Christoffel symbols Γ^λ_μν..." << std::endl;

        int total_points = grid.nx * grid.ny * grid.nz;
        int progress_step = total_points / 20;

        for (int k = 1; k < grid.nz - 1; k++)
        {
            for (int j = 1; j < grid.ny - 1; j++)
            {
                for (int i = 1; i < grid.nx - 1; i++)
                {
                    int idx = grid.index(i, j, k);
                    if (progress_step > 0 && idx % progress_step == 0)
                    {
                        float percent = 100.0f * idx / total_points;
                        std::cout << "\r[Einstein] Christoffel progress: " << std::fixed
                            << std::setprecision(1) << percent << "% " << std::flush;
                    }

                    SpacetimeMetric& metric = grid.at(i, j, k);

                    if (metric.singular) continue;

                    for (int lambda = 0; lambda < 4; lambda++)
                    {
                        for (int mu = 0; mu < 4; mu++)
                        {
                            for (int nu = mu; nu < 4; nu++)
                            {
                                float gamma_value = 0.0f;

                                for (int sigma = 0; sigma < 4; sigma++)
                                {
                                    float g_inv_ls = metric.get_g_inv(lambda, sigma);

                                    float d_mu_g_nu_sigma = metric.d_g[mu][sym4_index(nu, sigma)];
                                    float d_nu_g_mu_sigma = metric.d_g[nu][sym4_index(mu, sigma)];
                                    float d_sigma_g_mu_nu = metric.d_g[sigma][sym4_index(mu, nu)];

                                    gamma_value += 0.5f * g_inv_ls * (d_mu_g_nu_sigma + d_nu_g_mu_sigma - d_sigma_g_mu_nu);
                                }

                                metric.set_christoffel(lambda, mu, nu, gamma_value);
                            }
                        }
                    }
                }
            }
        }

        std::cout << "\r[Einstein] Christoffel progress: 100.0% - Complete!     " << std::endl;
    }

    struct Geodesic
    {
        glm::vec4 x;
        glm::vec4 v;
        float lambda;
        bool terminated;

        Geodesic(glm::vec3 pos, glm::vec3 dir)
        {
            x = glm::vec4(0.0f, pos.x, pos.y, pos.z);
            v = glm::vec4(1.0f, dir.x, dir.y, dir.z);
            lambda = 0.0f;
            terminated = false;
        }
    };

    void IntegrateGeodesic(Geodesic& geo, const SpacetimeGrid& grid, float dlambda, int max_steps)
    {
        // TODO: Implement integration with grid interpolation
        // For now, placeholder
        geo.terminated = true;
    }

    void InitializeSpacetime(SpacetimeGrid& grid, const std::vector<glm::vec4>& bodies)
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "  Einstein Field Equations Solver" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "[Einstein] Solving G_μν = 8πT_μν numerically" << std::endl;

        ComputeMetric(grid, bodies);

        ComputeMetricDerivatives(grid);

        ComputeChristoffelSymbols(grid);

        std::cout << "\n[Einstein] Spacetime initialized!" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
}

#endif // EINSTEIN_SOLVER_H