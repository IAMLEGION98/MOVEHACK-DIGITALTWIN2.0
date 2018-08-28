package com.example.anupam.test2;

import android.content.Intent;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CalendarView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Locale;

public class LeaveActivity2 extends AppCompatActivity {


    TextView tvdisp,nod;
    String gencity=null;
    CalendarView caltodate;
    Date dfrom,dto;
    DatabaseReference leavebase;
    Integer nodl=0;
    String datereg=null;
    String eid=null;
    Button confirm;
    int [] monthDays={31, 28, 31, 30, 31, 30,
            31, 31, 30, 31, 30, 31};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_leave2);

        Calendar cal = Calendar.getInstance();
         datereg =cal.getTime().toString();

        Log.e("Todays Date is :",cal.getTime().toString());
        Toast.makeText(getApplicationContext(),cal.getTime().toString(),Toast.LENGTH_SHORT).show();
        confirm = (Button)findViewById(R.id.btconfirm);
        tvdisp =(TextView)findViewById(R.id.tvupto);
        leavebase = FirebaseDatabase.getInstance().getReference("empleave");
        caltodate = (CalendarView)findViewById(R.id.calupto);

        Intent intent = getIntent();
        eid = intent.getStringExtra("employeeid");
        gencity = intent.getStringExtra("address");
        dfrom= new Date(intent.getIntExtra("dayf",0),intent.getIntExtra("monthf",0),intent.getIntExtra("yearf",0));
        nod = (TextView)findViewById(R.id.tvnod);
        caltodate.setOnDateChangeListener(new CalendarView.OnDateChangeListener() {
            @Override
            public void onSelectedDayChange(@NonNull CalendarView view, int year, int month, int dayOfMonth) {
                dto = new Date(dayOfMonth,month,year);
                String date = dayOfMonth+ "/"+month+"/"+year;
                tvdisp.setText(date);
                nodl = getDifference(dfrom,dto);
                nod.setText(nodl.toString());

            }
        });
        confirm.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(nodl>=0) {
                    addleave();
                    Handler handler = new Handler();
                    handler.postDelayed(new Runnable() {
                        @Override
                        public void run() {
                            Intent intentb = new Intent(getApplicationContext(),DashActivity.class);
                            startActivity(intentb);
                        }
                    },1300);


                }
                else
                    Toast.makeText(getApplicationContext(),"Please Select valid dates",Toast.LENGTH_SHORT).show();
            }
        });

    }

    public void addleave()
    {
        LeaveForm leaveForm1 = new LeaveForm(nodl,"no",gencity,datereg);
        leavebase.child(eid).setValue(leaveForm1);
        Toast.makeText(getApplicationContext(),"Leave Applied",Toast.LENGTH_LONG).show();

    }

    Integer getDifference(Date dt1, Date dt2)
    {
        // COUNT TOTAL NUMBER OF DAYS BEFORE FIRST DATE 'dt1'

        // initialize count using years and day
        Integer n1 = dt1.yy*365 + dt1.dd;

        // Add days for months in given date
        for (int i=0; i<dt1.mm - 1; i++)
            n1 += monthDays[i];

        // Since every leap year is of 366 days,
        // Add a day for every leap year
        n1 += countLeapYears(dt1);

        // SIMILARLY, COUNT TOTAL NUMBER OF DAYS BEFORE 'dt2'

        Integer n2 = dt2.yy*365 + dt2.dd;
        for (int i=0; i<dt2.mm - 1; i++)
            n2 += monthDays[i];
        n2 += countLeapYears(dt2);

        // return difference between two counts
        return (n2 - n1);
    }
    int countLeapYears(Date d)
    {
        int years = d.yy;

        // Check if the current year needs to be considered
        // for the count of leap years or not
        if (d.mm <= 2)
            years--;

        // An year is a leap year if it is a multiple of 4,
        // multiple of 400 and not a multiple of 100.
        return years / 4 - years / 100 + years / 400;
    }
}
