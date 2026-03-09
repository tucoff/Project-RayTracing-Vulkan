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
* CENÁRIOS DE ESPAÇO-TEMPO ESTACIONÁRIO
* 
* Estes cenários representam configurações estáticas de buracos negros
* para resolução das equações de Einstein no formalismo BSSN.
* 
* NOTA: Os corpos NÃO se movem. O espaço-tempo é calculado como solução
* estacionária das equações de campo de Einstein.
* 
* UNIDADES GEOMÉTRICAS (G = c = 1):
* - Massa: M (define o raio de Schwarzschild r_s = 2M)
* - Distância: Unidades de comprimento arbitrárias
* 
* MASSAS TÍPICAS:
* - Massa 1-5:   Buracos negros estelares pequenos (r_s = 2-10)
* - Massa 5-20:  Buracos negros estelares médios (r_s = 10-40)
* - Massa 20-50: Buracos negros estelares massivos (r_s = 40-100)
* - Massa 50+:   Buracos negros supermassivos (r_s > 100)
* 
* SEPARAÇÕES RECOMENDADAS:
* - Para lensing gravitacional visível: separação > 3 × (M1 + M2)
* - Para evitar horizontes comuns: separação > (r_s1 + r_s2)
* - Para sistema estável: separação > 5 × max(M_i)
*/
