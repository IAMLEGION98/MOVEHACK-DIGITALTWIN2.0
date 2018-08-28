package com.example.anupam.test2;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import org.w3c.dom.Text;

public class LeaveStatus extends AppCompatActivity {


    TextView message,days,approval;
    DatabaseReference statusbase;
    String eid =null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_leave_status);
        Intent mintent = getIntent();
        eid = mintent.getStringExtra("employeeid");

        statusbase = FirebaseDatabase.getInstance().getReference("empleave");
        message = (TextView)findViewById(R.id.tvmsg);
        days = (TextView)findViewById(R.id.tvdays);
        approval = (TextView)findViewById(R.id.tvapp);

        statusbase.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                int flag=0;
                for(DataSnapshot postsnapshot: dataSnapshot.getChildren())
                {
                    String idval = postsnapshot.getKey().toString();
                    LeaveForm form1 = postsnapshot.getValue(LeaveForm.class);
                    if(idval.equals(eid))
                    {
                        days.setText(""+form1.ndays);
                        message.setText("Your leave info is : ");
                        approval.setText(form1.approval);
                        flag=1;

                    }
                }
                if(flag==0)
                    message.setText("You dont have any awaiting leave approvals..");

            }

            @Override
            public void onCancelled(DatabaseError databaseError) {

            }
        });



    }
}
