/**
 * NCNN Deployment - C++ Reference Implementation
 * Target: ARM devices (RK3588, Jetson Nano, Raspberry Pi)
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DNCNN_ROOT=/path/to/ncnn
 *   make
 *
 * Run:
 *   ./ncnn_deploy best.param best.bin 0   # webcam
 *   ./ncnn_deploy best.param best.bin image.jpg
 *
 * Export NCNN format:
 *   python -c "
 *     from ultralytics import YOLO
 *     model = YOLO('best.pt')
 *     model.export(format='ncnn')
 *   "
 */

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

// NCNN headers (install: https://github.com/Tencent/ncnn)
#include <ncnn/net.h>
#include <ncnn/mat.h>

// OpenCV for image/video I/O
#include <opencv2/opencv.hpp>

struct Detection {
    int class_id;
    float confidence;
    float x1, y1, x2, y2;
};

static const char* CLASS_NAMES[] = {"recyclable", "hazardous", "kitchen", "other"};
static const cv::Scalar CLASS_COLORS[] = {
    cv::Scalar(0, 255, 0),     // green
    cv::Scalar(0, 0, 255),     // red
    cv::Scalar(255, 0, 0),     // blue
    cv::Scalar(128, 128, 128)  // gray
};

class YOLONCNN {
public:
    YOLONCNN(const char* param_path, const char* bin_path, int imgsz = 640,
             float conf_thresh = 0.25f, float iou_thresh = 0.45f)
        : imgsz_(imgsz), conf_thresh_(conf_thresh), iou_thresh_(iou_thresh)
    {
        net_.load_param(param_path);
        net_.load_model(bin_path);
        printf("[INFO] NCNN model loaded: %s / %s\n", param_path, bin_path);
    }

    std::vector<Detection> detect(const cv::Mat& bgr) {
        int orig_w = bgr.cols;
        int orig_h = bgr.rows;

        // Preprocess: resize + normalize
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
            bgr.cols, bgr.rows, imgsz_, imgsz_
        );

        const float mean_vals[3] = {0.f, 0.f, 0.f};
        const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        // Inference
        ncnn::Extractor ex = net_.create_extractor();
        ex.input("images", in);

        ncnn::Mat out;
        ex.extract("output0", out);

        // Post-process
        // output shape: (1, 84, 8400) for 4 classes
        // 84 = 4 (box) + 80 (coco) or 4 (box) + 4 (our classes)
        int num_dets = out.w;       // 8400
        int num_attrs = out.h;      // 84

        std::vector<Detection> results;
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        for (int i = 0; i < num_dets; i++) {
            const float* ptr = out.channel(0) + i * num_attrs;

            float cx = ptr[0];
            float cy = ptr[1];
            float w  = ptr[2];
            float h  = ptr[3];

            // Find best class
            int best_cls = 0;
            float best_score = ptr[4];
            for (int c = 1; c < num_attrs - 4; c++) {
                if (ptr[4 + c] > best_score) {
                    best_score = ptr[4 + c];
                    best_cls = c;
                }
            }

            if (best_score < conf_thresh_) continue;

            float x1 = (cx - w / 2.f) * orig_w / imgsz_;
            float y1 = (cy - h / 2.f) * orig_h / imgsz_;
            float x2 = (cx + w / 2.f) * orig_w / imgsz_;
            float y2 = (cy + h / 2.f) * orig_h / imgsz_;

            // Clip
            x1 = std::max(0.f, std::min(x1, (float)(orig_w - 1)));
            y1 = std::max(0.f, std::min(y1, (float)(orig_h - 1)));
            x2 = std::max(0.f, std::min(x2, (float)(orig_w - 1)));
            y2 = std::max(0.f, std::min(y2, (float)(orig_h - 1)));

            boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            scores.push_back(best_score);
            class_ids.push_back(best_cls);
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, conf_thresh_, iou_thresh_, indices);

        for (int idx : indices) {
            Detection det;
            det.class_id = class_ids[idx];
            det.confidence = scores[idx];
            det.x1 = boxes[idx].x;
            det.y1 = boxes[idx].y;
            det.x2 = boxes[idx].x + boxes[idx].width;
            det.y2 = boxes[idx].y + boxes[idx].height;
            results.push_back(det);
        }

        return results;
    }

    void draw(cv::Mat& frame, const std::vector<Detection>& dets) {
        for (const auto& det : dets) {
            cv::Scalar color = CLASS_COLORS[det.class_id % 4];
            cv::rectangle(frame,
                          cv::Point(det.x1, det.y1),
                          cv::Point(det.x2, det.y2),
                          color, 2);

            char label[64];
            snprintf(label, sizeof(label), "%s %.2f",
                     CLASS_NAMES[det.class_id % 4], det.confidence);

            int baseline = 0;
            cv::Size tsz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            cv::rectangle(frame,
                          cv::Point(det.x1, det.y1 - tsz.height - 6),
                          cv::Point(det.x1 + tsz.width, det.y1),
                          color, -1);
            cv::putText(frame, label,
                        cv::Point(det.x1, det.y1 - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
    }

private:
    ncnn::Net net_;
    int imgsz_;
    float conf_thresh_;
    float iou_thresh_;
};

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <param> <bin> <source>\n", argv[0]);
        printf("  source: 0 for webcam, or image/video path\n");
        return 1;
    }

    YOLONCNN detector(argv[1], argv[2]);

    std::string source = argv[3];

    // Try to parse as integer (webcam)
    char* end;
    int cam_id = strtol(source.c_str(), &end, 10);
    if (*end == '\0') {
        // Webcam
        cv::VideoCapture cap(cam_id);
        if (!cap.isOpened()) {
            printf("[ERROR] Cannot open webcam %d\n", cam_id);
            return 1;
        }

        printf("[INFO] Webcam opened, press 'q' to quit\n");
        cv::Mat frame;
        while (cap.read(frame)) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto dets = detector.detect(frame);
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

            detector.draw(frame, dets);

            char fps_text[32];
            snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", 1000.f / ms);
            cv::putText(frame, fps_text, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            cv::imshow("NCNN Garbage Detection", frame);
            if (cv::waitKey(1) == 'q') break;
        }
    } else {
        // Image file
        cv::Mat img = cv::imread(source);
        if (img.empty()) {
            printf("[ERROR] Cannot read image: %s\n", source.c_str());
            return 1;
        }

        auto dets = detector.detect(img);
        detector.draw(img, dets);

        printf("Detected %d objects:\n", (int)dets.size());
        for (const auto& d : dets) {
            printf("  %s %.3f [%.0f,%.0f,%.0f,%.0f]\n",
                   CLASS_NAMES[d.class_id % 4], d.confidence,
                   d.x1, d.y1, d.x2, d.y2);
        }

        cv::imshow("Detection", img);
        cv::waitKey(0);
    }

    return 0;
}
