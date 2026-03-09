#include "game.h"
#include <iostream>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

int main() 
{
#if defined(_WIN32) || defined(_WIN64)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    
    play();
    return 0;
}   