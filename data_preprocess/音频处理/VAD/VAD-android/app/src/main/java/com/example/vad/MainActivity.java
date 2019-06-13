package com.example.vad;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.media.MediaPlayer;
import android.media.SoundPool;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("libfvad");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button play1 = findViewById(R.id.play1);
        Button play2 = findViewById(R.id.play2);

        play1.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View v) {
                mediaPlayer("voice_1.wav");
            }
        });

        play2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                short [] audio_data = remove_silence("voice_1.wav");
                audiotrackPlayer(BytesTransUtil.getInstance().Shorts2Bytes(audio_data));
            }
        });

    }

    public short[] remove_silence(String filename){
        WebRTCVad vad = new WebRTCVad(8000,1,20);
        short []  audio = null;
        try {
            InputStream is = getAssets().open(filename);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            byte [] voice_buff = new byte[size-44];
            System.arraycopy(buffer,44,voice_buff,0,size-44);
            short [] voice_data = BytesTransUtil.getInstance().Bytes2Shorts(voice_buff);
            audio = vad.remove_silence(voice_data);
            Log.d("zhanjie","src length "+voice_data.length+" dest length "+audio.length);
        } catch (IOException e) {
            e.printStackTrace();
        }
        vad.releaseVAD();
        return audio;
    }



    public void audiotrackPlayer(String filename){
        try {
            InputStream is = getAssets().open(filename);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            int bufferSize = AudioTrack.getMinBufferSize(8000, AudioFormat.CHANNEL_CONFIGURATION_MONO, AudioFormat.ENCODING_PCM_16BIT);
            AudioTrack audioTrack = new AudioTrack(AudioManager.STREAM_MUSIC,
                    8000, AudioFormat.CHANNEL_CONFIGURATION_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize, AudioTrack.MODE_STREAM);
            audioTrack.play();
            audioTrack.write(buffer, 0, buffer.length);
            audioTrack.stop();
            audioTrack.release();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void audiotrackPlayer(byte [] audioData){
        int bufferSize = AudioTrack.getMinBufferSize(8000, AudioFormat.CHANNEL_CONFIGURATION_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioTrack audioTrack = new AudioTrack(AudioManager.STREAM_MUSIC,
                8000, AudioFormat.CHANNEL_CONFIGURATION_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize, AudioTrack.MODE_STREAM);
        audioTrack.play();
        audioTrack.write(audioData, 0, audioData.length);
        audioTrack.stop();
        audioTrack.release();
    }


    public void mediaPlayer(String filename){
        AssetManager assetManager = MainActivity.this.getAssets();
        try {
            AssetFileDescriptor afd = assetManager.openFd(filename);
            MediaPlayer player = new MediaPlayer();
            player.setDataSource(afd.getFileDescriptor(),afd.getStartOffset(), afd.getLength());
            //player.setLooping(true);
            player.prepare();
            player.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
