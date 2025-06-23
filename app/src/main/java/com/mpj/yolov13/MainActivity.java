package com.mpj.yolov13;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;

import com.mpj.yolov13.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

	// Used to load the 'yolov13' library on application startup.
	static {
		System.loadLibrary("yolov13");
	}

	private ActivityMainBinding binding;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		binding = ActivityMainBinding.inflate(getLayoutInflater());
		setContentView(binding.getRoot());

		// Example of a call to a native method
		TextView tv = binding.sampleText;
		tv.setText(stringFromJNI());
	}

	/**
	 * A native method that is implemented by the 'yolov13' native library,
	 * which is packaged with this application.
	 */
	public native String stringFromJNI();
}