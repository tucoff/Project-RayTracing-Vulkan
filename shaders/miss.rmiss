#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;

void main()
{
    // Cor azul para representar o céu
    hitValue = vec3(0.4, 0.6, 1.0);
}
