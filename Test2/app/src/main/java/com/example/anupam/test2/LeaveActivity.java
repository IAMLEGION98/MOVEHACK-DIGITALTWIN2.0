package com.example.anupam.test2;

import android.content.Intent;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CalendarView;
import android.widget.TextView;
import android.widget.Toast;

public class LeaveActivity extends AppCompatActivity {


    TextView datefrom;
    String gencity=null;
    CalendarView calfrom;
    Date dfrom;
    String eid =null;
    Button btgo;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_leave);
        btgo = (Button)findViewById(R.id.btnext);
        Intent intent2 = getIntent();
        eid = intent2.getStringExtra("employeeid");
        gencity = intent2.getStringExtra("address");
        datefrom=(TextView)findViewById(R.id.tvfrom);
        calfrom =(CalendarView)findViewById(R.id.calfrom1);

        calfrom.setOnDateChangeListener(new CalendarView.OnDateChangeListener() {
            @Override
            public void onSelectedDayChange(@NonNull CalendarView view, int year, int month, int dayOfMonth) {
                dfrom = new Date(dayOfMonth,month,year);
                String date = dayOfMonth+ "/"+month+"/"+year;
                datefrom.setText(date);

            }
        });
        btgo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(dfrom==null)
                {
                    Toast.makeText(getApplicationContext(),"NO date picked",Toast.LENGTH_SHORT).show();
                }
               else
                {
                    Intent mintent = new Intent(getApplicationContext(),LeaveActivity2.class);
                    mintent.putExtra("dayf",dfrom.dd);
                    mintent.putExtra("monthf",dfrom.mm);
                    mintent.putExtra("yearf",dfrom.yy);
                    mintent.putExtra("address",gencity);
                    mintent.putExtra("employeeid",eid);
                    startActivity(mintent);

                }
            }
        });
    }
}
