#!/bin/bash
GLSLC=/usr/bin/glslc
$GLSLC shaders/shader.vert -o shaders/vert.spv
$GLSLC shaders/shader.frag -o shaders/frag.spv