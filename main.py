import glob
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np


def create_video(output_folder, image_files):
    """画像ファイルから動画を作成"""
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape
    frame_size = (width, height)
    fps = 24

    # 動画ファイルの出力先
    output_file = 'output.mov'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(output_folder,output_file), fourcc, fps, frame_size)

    for image_file in image_files:
        frame = cv2.imread(image_file)
        video_writer.write(frame)

    video_writer.release()
    print(f'動画ファイル {output_file} を作成しました。')


if __name__ == '__main__':

    image_files = []
    output_folder = ''
    for path in glob.glob('hotpixel/*.exr'):
        output_folder = os.path.join(os.path.dirname(path), 'check')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(path))[0] + '.jpg')
        image_files.append(output_path)

        # 画像を読み込み、グレースケールに変換
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE|cv2.IMREAD_UNCHANGED)

        # メディアンフィルターを適用
        median = cv2.medianBlur(img, 3)

        # 元画像とメディアンフィルター適用画像の差分を計算
        diff = cv2.absdiff(img, median)

        # 差分が閾値以上のピクセルをホットピクセルと判定
        _, hot_pixels = cv2.threshold(diff, 30.0/255.0, 1.0, cv2.THRESH_BINARY)
        # ホットピクセルの位置を取得
        coords_hot = np.column_stack(np.where(hot_pixels > 0))

        print(f"ホットピクセルの数: {len(coords_hot)}")
        cv2.putText(img, os.path.basename(path), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0.0, 0.0, 1.0), 2,
                    cv2.LINE_AA)
        cv2.putText(img, f"num of pix: {len(coords_hot)}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0.0, 0.0, 1.0), 2,
                    cv2.LINE_AA)
        for (y, x, _) in coords_hot:
            print(f"x:{x}, y:{y}")
            cv2.circle(img, (x, y), 50, (0, 0, 1.0), 5)

        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(output_path, img_uint8)

    create_video(output_folder,image_files)