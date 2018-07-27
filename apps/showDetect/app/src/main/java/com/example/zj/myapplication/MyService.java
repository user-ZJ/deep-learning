package com.example.zj.myapplication;

import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.RectF;
import android.os.IBinder;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;

public class MyService extends Service {
    final static String TAG="jack";
    MyView mv = new MyView(MyService.this);

    public MyService() {
        mv.showDetect();
    }

    @Override
    public IBinder onBind(Intent intent) {
        // TODO: Return the communication channel to the service.
        throw new UnsupportedOperationException("Not yet implemented");
    }

    private class MyView extends View{
        Paint mpaint;

        public MyView(Context context) {
            super(context);
            mpaint = new Paint();
        }


        @Override
        public void onDraw(Canvas canvas) {

            mpaint.setColor(Color.RED);
            mpaint.setStyle(Paint.Style.STROKE);
            mpaint.setStrokeWidth(2.0f);
            canvas.drawRect(new RectF(100.0f,100.0f,300.0f,300.0f), mpaint);
        }

        public void showDetect(){
            Context mContext = MyService.this;
            WindowManager mWindowManager = (WindowManager) mContext
                    .getSystemService(Context.WINDOW_SERVICE);
            WindowManager.LayoutParams params = new WindowManager.LayoutParams();
            params.type = WindowManager.LayoutParams.TYPE_SYSTEM_ALERT;
            int flags = WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL;
            params.flags = flags;
            params.format = PixelFormat.TRANSLUCENT;
            params.width = WindowManager.LayoutParams.MATCH_PARENT;
            params.height = WindowManager.LayoutParams.MATCH_PARENT;
            params.gravity = Gravity.CENTER;

        }
    }
}
