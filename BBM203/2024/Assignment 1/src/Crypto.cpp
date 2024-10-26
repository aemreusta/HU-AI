#include "Crypto.h"
#include "GrayscaleImage.h"


// Extract the least significant bits (LSBs) from SecretImage, calculating x, y based on message length
std::vector<int> Crypto::extract_LSBits(SecretImage& secret_image, int message_length) {
    std::vector<int> LSB_array;

    GrayscaleImage image = secret_image.reconstruct();
    int width = image.get_width();
    int height = image.get_height();
    
    int total_bits = message_length * 7;
    
    if (width * height < total_bits) {
        throw std::runtime_error("Image does not have enough pixels to store the message.");
    }
    
    int start_pixel = width * height - total_bits;
    
    for (int i = start_pixel; i < width * height; ++i) {
        int x = i % width;
        int y = i / width;
        int pixel_value = image.get_pixel(y, x);
        LSB_array.push_back(pixel_value & 1);  
    }

    return LSB_array;
}

// Decrypt message by converting LSB array into ASCII characters
std::string Crypto::decrypt_message(const std::vector<int>& LSB_array) {
    std::string message;
    
    if (LSB_array.size() % 7 != 0) {
        throw std::runtime_error("LSB array size is not a multiple of 7.");
    }

    // 2. Her 7 bit grubunu ASCII karakterine dönüştür
    for (size_t i = 0; i < LSB_array.size(); i += 7) {
        int char_code = 0;
        for (int j = 0; j < 7; ++j) {
            char_code = (char_code << 1) | LSB_array[i + j];
        }
        message += static_cast<char>(char_code);
    }

    return message;
}

// Encrypt message by converting ASCII characters into LSBs
std::vector<int> Crypto::encrypt_message(const std::string& message) {
    std::vector<int> LSB_array;

    for (char c : message) {
        std::bitset<7> bits(c);
        for (int i = 6; i >= 0; --i) {
            LSB_array.push_back(bits[i]);
        }
    }

    return LSB_array;
}

// Embed LSB array into GrayscaleImage starting from the last bit of the image
SecretImage Crypto::embed_LSBits(GrayscaleImage& image, const std::vector<int>& LSB_array) {
    int width = image.get_width();
    int height = image.get_height();

    if (width * height < LSB_array.size()) {
        throw std::runtime_error("Image does not have enough pixels to store the LSB array.");
    }
    
    int start_pixel = width * height - LSB_array.size();

    for (size_t i = 0; i < LSB_array.size(); ++i) {
        int x = (start_pixel + i) % width;
        int y = (start_pixel + i) / width;
        
        int pixel_value = image.get_pixel(y, x);
        pixel_value = (pixel_value & ~1) | LSB_array[i]; 
        image.set_pixel(y, x, pixel_value);
    }

    return SecretImage(image);
}
