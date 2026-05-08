std_msgs/Header header
float32 center        # normalized ego-lane center x [0.0, 1.0]; -1.0 = no detection
float32 confidence    # detection confidence [0.0, 1.0]; 0.0 = N/A (pure vision)
bool    detected      # true when lane is actively detected / tracked
float32 lx            # left lane x at y_ref (pixels); -1.0 = not detected
float32 rx            # right lane x at y_ref (pixels); -1.0 = not detected
