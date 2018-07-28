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
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;

import java.util.Timer;
import java.util.TimerTask;

public class MyService extends Service {
    final static String TAG="jack";
    private MyView mv;
    private Timer timer;
    private TimerTask task;


    public void onCreate() {
        super.onCreate();
        Log.w(TAG, "in onCreate");

    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.w(TAG, "MyService:" + this);
        mv = new MyView(MyService.this);
        timer = new Timer();
        task = new TimerTask(){

            @Override
            public void run() {
                //mv.showDetect();
            }
        };
        timer.schedule(task,0,1);

        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.w(TAG, "in onDestroy");
        mv.removeDetect();
    }

    @Override
    public IBinder onBind(Intent intent) {
        // TODO: Return the communication channel to the service.
        throw new UnsupportedOperationException("Not yet implemented");
    }

    private class MyView extends View{
        MyView mView;


        public MyView(Context context) {
            super(context);
            mView = this;
            WindowManager mWindowManager = (WindowManager) getApplicationContext()
                    .getSystemService(Context.WINDOW_SERVICE);
            WindowManager.LayoutParams params = new WindowManager.LayoutParams();
            params.type = WindowManager.LayoutParams.TYPE_SYSTEM_ALERT;
            int flags = WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL;
            params.flags = flags;
            params.format = PixelFormat.TRANSLUCENT;
            params.width = WindowManager.LayoutParams.MATCH_PARENT;
            params.height = WindowManager.LayoutParams.MATCH_PARENT;
            params.gravity = Gravity.CENTER;
            mWindowManager.addView(mView,params);
        }


        @Override
        public void onDraw(Canvas canvas) {
            Paint mpaint = new Paint();
            mpaint.setColor(Color.RED);
            mpaint.setStyle(Paint.Style.STROKE);
            mpaint.setStrokeWidth(2.0f);
            canvas.drawRect(new RectF(100.0f,100.0f,300.0f,300.0f), mpaint);
        }

        public void showDetect(){
            invalidate();

        }

        public void removeDetect(){
            WindowManager mWindowManager = (WindowManager) getApplicationContext()
                    .getSystemService(Context.WINDOW_SERVICE);
            WindowManager.LayoutParams params = new WindowManager.LayoutParams();
            params.type = WindowManager.LayoutParams.TYPE_SYSTEM_ALERT;
            params.flags = WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL;

            mWindowManager.removeViewImmediate(mView);
        }
    }
}
