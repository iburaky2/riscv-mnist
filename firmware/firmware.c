#include <limits.h>
#include <stdint.h>
#include "../data/weights.h"

#define CAMERA_CAPTURE ((volatile uint32_t*)0x20000000)
#define CAMERA_READ    ((volatile uint32_t*)0x20000004)
#define RESULT_WRITE   ((volatile uint32_t*)0x20000008)

int main() {
  uint8_t image[784];

  // Test 100 images.
  for (int i = 0; i < 100; i++) {
    // Capture new image.
    *CAMERA_CAPTURE = 1;
  
    // Read pixels from the camera.
    for (int i = 0; i < 784; i++) {
      image[i] = *CAMERA_READ;
    }
  
    // Run classification.
    int32_t total[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      total[i] = 0;
      for (int j = 0; j < INPUT_SIZE; j++) {
        total[i] += image[j] * WEIGHTS[i][j];
      }
      total[i] += BIASES[i];
    }

    int32_t digit = 0;
    int32_t digit_total = INT32_MIN;

    for (int i = 0; i < OUTPUT_SIZE; i++) {
      if (digit_total < total[i]) {
        digit_total = total[i];
        digit = i;
      }
    }

    // Write the result.
    *RESULT_WRITE = digit;
  }

  return 0;
}

void _start() __attribute__((section(".init")));
void _start() {
  main();
  while(1); // Halt.
}
