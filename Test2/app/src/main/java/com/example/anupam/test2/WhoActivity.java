package com.example.anupam.test2;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.CardView;
import android.view.View;

public class WhoActivity extends AppCompatActivity {

    CardView employee,manager;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_who);

        employee =(CardView)findViewById(R.id.empid);
        manager = (CardView)findViewById(R.id.managerid);

        employee.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(WhoActivity.this,MainActivity.class);
                startActivity(intent);
                finish();
            }
        });

        manager.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
             Intent intent2 = new Intent(WhoActivity.this,PendingRequests.class);
             startActivity(intent2);
             finish();
            }
        });


    }
}
