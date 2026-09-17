#pragma once

#include <chrono>

struct timer {
    timer() {
        start();
        stop();
    }

    void start() {
        t_beg = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        t_end = std::chrono::high_resolution_clock::now();
    }

    size_t elapsed() {
      return static_cast<size_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_beg).count());
    }

    std::chrono::high_resolution_clock::time_point t_beg;
    std::chrono::high_resolution_clock::time_point t_end;
};
