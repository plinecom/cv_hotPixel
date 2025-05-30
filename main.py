# これはサンプルの Python スクリプトです。

def print_hi(name):
    # スクリプトをデバッグするには以下のコード行でブレークポイントを使用してください。
    print(f'Hi, {name}')  # Ctrl+F8を押すとブレークポイントを切り替えます。
import glob
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
# ガター内の緑色のボタンを押すとスクリプトを実行します。
if __name__ == '__main__':



    for path in glob.glob('hotpixel/*.exr'):
        output_folder = os.path.join(os.path.dirname(path), 'check')
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(path))[0] + '.jpg')

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
        cv2.putText(img, f"num of pix: {len(coords_hot)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0.0, 0.0, 1.0), 2,
                    cv2.LINE_AA)
        for (x, y, v) in coords_hot:
            print(f"x:{y}, y:{x}")
            cv2.circle(img, (y, x), 50, (0, 0, 1.0), 5)

        # 結果を表示
        '''
        cv2.imshow("Hot Pixels", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(output_path, img_uint8)
    #    print_hi('PyCharm')

# PyCharm のヘルプは https://www.jetbrains.com/help/pycharm/ を参照してください
