#pragma once
#include "opencv2/imgproc.hpp"
#include <vector>
#include <string>

class __declspec(dllexport) Helpers
{
public:
    static std::vector<float> loadImage(const std::string& filename, int sizeX = 64, int sizeY = 64);
    static std::vector<float> loadImageRGB(const std::string& imageFile, int width = 224, int height = 224);

    static std::vector<std::string> loadLabels(const std::string& filename);
};