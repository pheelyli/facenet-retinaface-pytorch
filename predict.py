#----------------------------------------------------#
#   对视频中的predict.py进行了修改，
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#   python predict.py /temp1/img /temp1/result  
#   python predict.py N:\backupSync\DigitalPhoto\2021 w:\temp
#----------------------------------------------------#
import time
import os, sys
import cv2
import numpy as np

from retinaface import Retinaface

if __name__ == "__main__":
    retinaface = Retinaface()
    #-------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测
    #   'video'表示视频检测
    #   'fps'表示测试fps
    #-------------------------------------------------------------------------#
    mode = "batch"
    #-------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出才会完成完整的保存步骤，不可直接结束程序。
    #-------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    if mode == "batch":
        '''
        predict.py有几个注意点
        1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用cv2.imread打开图片文件进行预测。
        2、如果想要保存，利用cv2.imwrite("img.jpg", r_image)即可保存。
        3、如果想要获得框的坐标，可以进入detect_image函数，读取(b[0], b[1]), (b[2], b[3])这四个值。
        4、如果想要截取下目标，可以利用获取到的(b[0], b[1]), (b[2], b[3])这四个值在原图上利用矩阵的方式进行截取。
        5、在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
        '''
        if len(sys.argv) == 3:
            srcDir = sys.argv[1]
            dstDir = sys.argv[2]
        else:
            srcDir = input('Input source dir:')
            dstDir = input('Input destination dir:')

        for root, dirs, files in os.walk( srcDir):
            for f in files:
                file =os.path.join(root,f)
                # file = file.replace("\\","/"  )
                ext = os.path.splitext(file)[1]
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    img = os.path.join(srcDir,file)
                    result = os.path.join(dstDir,file)
                    print (img)
                    # image = cv2.imread(img)
                    # 因为imread不支持中文文件名，所以需要由open读了文件后再交给np
                    stream = open(img, "rb")
                    bytes = bytearray(stream.read())
                    numpyarray = np.asarray(bytes, dtype=np.uint8)
                    image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

                    image   = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    r_image = retinaface.detect_image(image)

                    # r_image = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(result, r_image)

    elif  mode == "predict":
        '''
        predict.py有几个注意点
        1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用cv2.imread打开图片文件进行预测。
        2、如果想要保存，利用cv2.imwrite("img.jpg", r_image)即可保存。
        3、如果想要获得框的坐标，可以进入detect_image函数，读取(b[0], b[1]), (b[2], b[3])这四个值。
        4、如果想要截取下目标，可以利用获取到的(b[0], b[1]), (b[2], b[3])这四个值在原图上利用矩阵的方式进行截取。
        5、在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
        '''
        while True:
            img = input('Input image filename:')
            image = cv2.imread(img)
            if image is None:
                print('Open Error! Try again!')
                continue
            else:
                image   = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                r_image = retinaface.detect_image(image)
                r_image = cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR)
                cv2.imshow("after",r_image)
                cv2.waitKey(0)

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref,frame=capture.read()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 进行检测
            frame = np.array(retinaface.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                    
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 100
        img = cv2.imread('img/obama.jpg')
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        tact_time = retinaface.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")
