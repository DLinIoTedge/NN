package c.r.fighte.j7;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.net.Uri;
import android.nfc.Tag;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.apache.commons.io.IOUtils;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;
import java.util.ArrayList;

class MView extends View {
    Paint mPaint;
    Context c;
    float mX, mY;
    Path mpath;
    private Bitmap mbit;
    private Canvas mcanvas;
    ArrayList<Path> paths = new ArrayList<Path>();

    //Converts dp to pixel, so that image is proper size on screen
    public int dpToPx(int dp) {
        return (int) (dp * Resources.getSystem().getDisplayMetrics().density);
    }

    public MView(Context context) {
        super(context);

        mbit = Bitmap.createBitmap(dpToPx(224), dpToPx(224), Bitmap.Config.ARGB_8888);
        mcanvas = new Canvas(mbit);
        c = context;
        mPaint = new Paint();
        mPaint.setAntiAlias(false);
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setColor(Color.BLUE);
        mPaint.setStrokeWidth(30);
    }

    private void tStart(float x, float y) {
        mpath = new Path();
        paths.add(mpath);

        mpath.reset();
        mpath.moveTo(x, y);
        mX = x; mY = y;
    }

    private void tMove(float x, float y) {
        float dx = Math.abs(x - mX);
        float dy = Math.abs(y - mY);
        if(dx >= 1 || dy >= 1) {
            mpath.quadTo(mX, mY, 0.5f * (x + mX), 0.5f * (y + mY));
            mX = x;
            mY = y;
        }
    }

    private void tUp() {
        mpath.lineTo(mX, mY);
    }

    public Bitmap getBit() {
        return mbit;
    }

    public void clearBit() {
        mbit.eraseColor(Color.TRANSPARENT);
        paths.clear();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        canvas.save();
        for(Path p : paths) {
            mcanvas.drawPath(p, mPaint);
        }
        canvas.drawBitmap(mbit, 0, 0, null);
        canvas.restore();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                tStart(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_MOVE:
                tMove(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_UP:
                tUp();
                invalidate();
                break;
        }
        return true;
    }
}

public class MainActivity extends AppCompatActivity {
    public int start_offset;

    public DecimalFormat form = new DecimalFormat("##.####");
    public InputStream model_file;
    public boolean status = false;

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    private void setupNetworkFromBuffer(ByteBuffer netBuf) {
        //Network Vars
        int num_layers = netBuf.getInt();
        int layers[] = new int[num_layers];
        int num_weights = 0, num_bias = 0;
        for(int i = 0; i < num_layers; i++) {
            layers[i] = netBuf.getInt();
            if(i == 0) {
                num_weights += 784 * layers[i];
            } else {
                num_weights += layers[i - 1] * layers[i];
            }
            num_bias += layers[i];
        }
        float weights[] = new float[num_weights];
        float bias[] = new float[num_bias];
        int l = 0, m = 0;
        for(int i = 0; i < num_layers; i++) {
            int k_lim = i == 0 ? 784 : layers[i - 1];
            for(int j = 0; j < layers[i]; j++) {
                for(int k = 0; k < k_lim; k++) {
                    weights[l++] = netBuf.getFloat();
                }
                bias[m++] = netBuf.getFloat();
            }
        }
        initNet(num_layers, layers, bias, weights);
    }

    private void displayImage(Bitmap bm, TextView tv) {
        int img_size = bm.getWidth() * bm.getHeight();
        float data[] = new float[img_size];
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < bm.getHeight(); i++) {
            for(int j = 0; j < bm.getWidth(); j++) {
                boolean stat = bm.getPixel(j, i) == Color.BLUE;
                data[(i * bm.getWidth()) + j] = stat ? 1 : 0;
            }
        }
        float activ[] = inference(data);
        float total = 0;
        for(float f : activ) {
            total += f;
        }
        sb.append("\n");
        sb.append("       Activation         Percentage\n");
        for(int i = 0 ; i < 10; i++) {
            sb.append("->" + i + ":   " + form.format(activ[i]) + "      =>     " + form.format(activ[i] * 100 / total) + "%\n");
        }
        tv.setText(sb);

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        try {
            if (requestCode == 1818) {
                if (resultCode == RESULT_OK) {
                    model_file = getContentResolver().openInputStream(data.getData());
                    byte model_arr[] = IOUtils.toByteArray(model_file);
                    ByteBuffer buf = ByteBuffer.wrap(model_arr);
                    buf.order(ByteOrder.LITTLE_ENDIAN);
                    setupNetworkFromBuffer(buf);
                }
            }
        } catch (FileNotFoundException e) {Log.d("J722", "File not found");}
          catch (IOException e) {Log.d("J722", "Stream Error");}
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final LinearLayout ll = findViewById(R.id.llayout);
        final MView mv = new MView(this);
        Button nload = findViewById(R.id.netload);
        Button ldr = findViewById(R.id.ldr);
        ll.addView(mv);
        final TextView tv = findViewById(R.id.sample_text);
        Button goop = findViewById(R.id.doops);
        tv.setText(stringFromJNI());

        class getFile extends AsyncTask<Void, Void, Void> {
            @Override
            protected Void doInBackground(Void... voids) {
                try {
                    ServerSocket ssock = new ServerSocket(9090);
                    Socket sock = ssock.accept();
                    DataInputStream dis = new DataInputStream(sock.getInputStream());
                    byte net_arr[] = IOUtils.toByteArray(dis);
                    ByteBuffer buf = ByteBuffer.wrap(net_arr);
                    buf.order(ByteOrder.LITTLE_ENDIAN);
                    setupNetworkFromBuffer(buf);
                }catch (IOException e){}
                return null;
            }

            @Override
            protected void onPreExecute() {
                tv.setText("Waiting for file");
            }

            @Override
            protected void onPostExecute(Void v) {
                tv.setText("Received");
            }
        }

        try {
            InputStream net = getResources().openRawResource(R.raw.accurate);
            byte netArr[] = IOUtils.toByteArray(net);
            ByteBuffer netBuf = ByteBuffer.wrap(netArr);
            netBuf.order(ByteOrder.LITTLE_ENDIAN);

            setupNetworkFromBuffer(netBuf);

            ldr.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                    intent.setType("*/*");
                    intent.addCategory(Intent.CATEGORY_OPENABLE);
                    startActivityForResult(intent, 1818);
                }
            });

            nload.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    getFile fileTask = new getFile();
                    fileTask.execute();
                }
            });

            goop.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    Bitmap bm = Bitmap.createScaledBitmap(mv.getBit(), 28, 28, true);
                    displayImage(bm, tv);
                    mv.clearBit();
                }
            });
        } catch (IOException e) {
            tv.setText(e.getMessage());
        }
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    public native void initNet(int num_layers, int[] layers, float[] b, float[] weight);

    public native float[] inference(float img[]);
}
