#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>

#include "Helpers.h"

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>


std::vector<float> Helpers::loadImage(const std::string& filename, int sizeX, int sizeY)
{
    // Загружаем изображение в градациях серого (1 канал)
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "No image found at: " << filename << std::endl;
        return {};
    }

    // Изменяем размер изображения на 64x64
    cv::resize(image, image, cv::Size(sizeX, sizeY));

    // Преобразуем изображение в формат float и нормализуем значения в диапазон [0, 1]
    image.convertTo(image, CV_32F, 1.0 / 255);

    // Преобразуем изображение в 1D вектор
    return std::vector<float>(image.begin<float>(), image.end<float>());
}

// Загрузка RGB изображения (например, для API Model)
std::vector<float> Helpers::loadImageRGB(const std::string& imageFile, int width, int height) {
    cv::Mat image = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imageFile << std::endl;
        return {};
    }

    // Изменяем размер изображения
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(width, height));

    // Нормализация изображения (в пределах [0, 1])
    resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

    // Преобразуем в одномерный массив и меняем каналы на формат (C, H, W)
    std::vector<float> imageVec;
    for (int row = 0; row < resizedImage.rows; ++row) {
        for (int col = 0; col < resizedImage.cols; ++col) {
            cv::Vec3f& color = resizedImage.at<cv::Vec3f>(row, col);
            imageVec.push_back(color[2]);  // R
            imageVec.push_back(color[1]);  // G
            imageVec.push_back(color[0]);  // B
        }
    }
    return imageVec;
}



std::vector<std::string>Helpers::loadLabels(const std::string& filename)
{
    std::vector<std::string> output;

    std::ifstream file(filename);
    if (file) {
        std::string s;
        while (getline(file, s)) {
            output.emplace_back(s);
        }
        file.close();
    }

    return output;
}