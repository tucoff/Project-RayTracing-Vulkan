// scenarios.h - Cenários pré-configurados de sistemas de múltiplos corpos
// Inclua este arquivo em game.cpp caso queira usar cenários prontos

#ifndef SCENARIOS_H
#define SCENARIOS_H

#include <glm/glm.hpp>
#include <vector>

namespace Scenarios
{
    // Cenário 1: Buraco negro único (equivalente ao código original)
    std::vector<glm::vec4> SingleBlackHole()
    {
        return {
            glm::vec4(0.0f, 0.0f, -150.0f, 10.0f) // Posição (x,y,z), Massa
        };
    }

    // Cenário 2: Sistema binário simétrico
    std::vector<glm::vec4> BinarySymmetric()
    {
        return {
            glm::vec4(-20.0f, 0.0f, -150.0f, 10.0f),
            glm::vec4(20.0f, 0.0f, -150.0f, 10.0f)
        };
    }

    // Cenário 3: Sistema binário assimétrico (tipo merger)
    std::vector<glm::vec4> BinaryMerger()
    {
        return {
            glm::vec4(-15.0f, 0.0f, -150.0f, 20.0f), // Primário massivo
            glm::vec4(10.0f, 0.0f, -150.0f, 5.0f)    // Secundário menor
        };
    }

    // Cenário 4: Sistema triplo
    std::vector<glm::vec4> TripleSystem()
    {
        return {
            glm::vec4(0.0f, 0.0f, -150.0f, 15.0f),    // Centro
            glm::vec4(-30.0f, 0.0f, -150.0f, 8.0f),   // Esquerda
            glm::vec4(30.0f, 0.0f, -150.0f, 8.0f)     // Direita
        };
    }

    // Cenário 5: Cluster de 5 corpos
    std::vector<glm::vec4> Cluster5()
    {
        return {
            glm::vec4(0.0f, 0.0f, -150.0f, 12.0f),      // Centro
            glm::vec4(25.0f, 0.0f, -150.0f, 6.0f),      // Direita
            glm::vec4(-25.0f, 0.0f, -150.0f, 6.0f),     // Esquerda
            glm::vec4(0.0f, 25.0f, -150.0f, 6.0f),      // Cima
            glm::vec4(0.0f, -25.0f, -150.0f, 6.0f)      // Baixo
        };
    }

    // Cenário 6: "Galáxia" - anel de buracos negros
    std::vector<glm::vec4> Ring8()
    {
        std::vector<glm::vec4> bodies;
        const float radius = 40.0f;
        const int count = 8;
        const float mass = 5.0f;
        
        for (int i = 0; i < count; i++)
        {
            float angle = (2.0f * 3.14159f * i) / count;
            float x = radius * cos(angle);
            float z = -150.0f + radius * sin(angle);
            bodies.push_back(glm::vec4(x, 0.0f, z, mass));
        }
        
        // Adiciona um corpo massivo no centro
        bodies.push_back(glm::vec4(0.0f, 0.0f, -150.0f, 20.0f));
        
        return bodies;
    }

    // Cenário 7: Plano galáctico (distribuição planar)
    std::vector<glm::vec4> GalacticPlane()
    {
        std::vector<glm::vec4> bodies;
        
        // Grid 3x3 de buracos negros
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                if (i == 0 && j == 0) continue; // Pula o centro
                
                float x = i * 40.0f;
                float z = -150.0f + j * 40.0f;
                float mass = 5.0f + (rand() % 5); // Massas variadas
                
                bodies.push_back(glm::vec4(x, 0.0f, z, mass));
            }
        }
        
        return bodies;
    }

    // Cenário 8: Efeito de lente gravitacional extrema
    std::vector<glm::vec4> GravitationalLens()
    {
        return {
            glm::vec4(0.0f, 0.0f, -150.0f, 30.0f) // Um corpo super-massivo
        };
    }
}

#endif // SCENARIOS_H

/* 
 * EXEMPLO DE USO em game.cpp:
 * 
 * #include "scenarios.h"
 * 
 * void createBodiesStorageBuffer()
 * {
 *     // Escolha um cenário
 *     celestialBodies = Scenarios::BinaryMerger();
 *     
 *     // Ou crie manualmente
 *     // celestialBodies.clear();
 *     // celestialBodies.push_back(glm::vec4(x, y, z, mass));
 *     
 *     VkDeviceSize bufferSize = sizeof(glm::vec4) * celestialBodies.size();
 *     createBuffer(...);
 *     // ... resto do código
 * }
 * 
 * DICAS DE MASSAS (unidades geométricas):
 * - Massa 1-5:   Buracos negros estelares pequenos
 * - Massa 5-20:  Buracos negros estelares típicos
 * - Massa 20-50: Buracos negros estelares massivos
 * - Massa 50+:   Buracos negros intermediários/supermassivos
 * 
 * DICAS DE DISTÂNCIAS:
 * - Para visualizar lensing claro, mantenha separação > 2 × (M1 + M2)
 * - Horizontes não devem se sobrepor: dist > (M1 + M2) / 2
 * - Para cenários estáveis, use distâncias > 5 × max(M_i)
 */
