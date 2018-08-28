package com.example.anupam.test2;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ListView;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.ArrayList;
import java.util.List;

public class PendingRequests extends AppCompatActivity {

DatabaseReference userbase;
ListView listviewusers;
List<LeaveForm> userslist;
List userids;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pending_requests);

        userslist = new ArrayList<>();
        userids = new ArrayList();
        listviewusers = (ListView)findViewById(R.id.userlist);
        userbase = FirebaseDatabase.getInstance().getReference("empleave");



    }
    @Override
    protected void onStart() {
        super.onStart();
        userbase.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                userslist.clear();
                userids.clear();

                for(DataSnapshot usersnapshot:dataSnapshot.getChildren())
                {
                    userids.add(usersnapshot.getKey().toString());
                    LeaveForm lform = usersnapshot.getValue(LeaveForm.class);
                   userslist.add(lform);

                }
                Userlist adapter = new Userlist(PendingRequests.this,userslist,userids);
                listviewusers.setAdapter(adapter);
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {

            }
        });
    }

}
