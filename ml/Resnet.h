#pragma once
#include "IModel.h"
#include <onnxruntime_cxx_api.h>
#include <unordered_map>
#include <string>
#include "Helpers.h"

class ResnetModel : public IModel {
public:
    // Функция, которую будем экспортировать
    MODEL_API void RunModel(const std::string& imageFile, const std::string& labelFile, const std::wstring& modelPath) override;
};