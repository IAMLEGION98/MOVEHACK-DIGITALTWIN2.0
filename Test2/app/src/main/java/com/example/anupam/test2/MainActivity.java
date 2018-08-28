package com.example.anupam.test2;

import android.content.Intent;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Toast;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;


public class MainActivity extends AppCompatActivity {


    EditText username,password;
    Button add,signin;
    int val=0;
    ProgressBar pbar;
    String usersid,passid;
    DatabaseReference userbase;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        userbase = FirebaseDatabase.getInstance().getReference("Employees");
        pbar = (ProgressBar)findViewById(R.id.progressbar);

        add = (Button)findViewById(R.id.btadd);
        signin=(Button)findViewById(R.id.btsign);
        username = (EditText)findViewById(R.id.etid);
        password= (EditText)findViewById(R.id.etpass);
        add.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                adduser();
            }
        });
        signin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                searchusers();
            }
        });



    }
    public void adduser()
    {

        String user1 = username.getText().toString().trim();
        String pass1 = password.getText().toString().trim();
        if(!TextUtils.isEmpty(user1))
        {
            String id=userbase.push().getKey();
            Users newuser= new Users(user1,pass1);
            userbase.child(id).setValue(newuser);
            Toast.makeText(this,"User has been added",Toast.LENGTH_LONG).show();

        }
        else
        {
            Toast.makeText(this,"Please input",Toast.LENGTH_LONG).show();
        }
    }
    public void searchusers()
    {
        usersid= username.getText().toString().trim();
        passid = password.getText().toString().trim();
        if(!TextUtils.isEmpty(usersid)) {
            userbase.addValueEventListener(new ValueEventListener() {
                @Override
                public void onDataChange(DataSnapshot dataSnapshot) {
                    int flag = 0;
                    String fpass = null;
                    for (DataSnapshot postsnapshot : dataSnapshot.getChildren()) {
                        String idval = postsnapshot.getKey().toString();
                        Log.e("Id is this YOYO :",idval);
                        Users user = postsnapshot.getValue(Users.class);
                        pbar.setProgress(val+=10);
                        if (user.userid.equals(usersid) && user.password.equals(passid)) {
                            flag = 1;
                            pbar.setProgress(100);
                            Toast.makeText(getApplicationContext(), "Successfull match", Toast.LENGTH_SHORT).show();
                            Intent mintent = new Intent(MainActivity.this,DashActivity.class);
                            mintent.putExtra("employeeid",usersid);
                            mintent.putExtra("Userid",idval);
                            startActivity(mintent);
                            break;
                        }

                    }

                            if (flag == 0)
                                Toast.makeText(getApplicationContext(), "Invalid UserID/password", Toast.LENGTH_LONG).show();



                }

                @Override
                public void onCancelled(DatabaseError databaseError) {

                }
            });

        }
        else
            Toast.makeText(getApplicationContext(),"Please input details",Toast.LENGTH_SHORT).show();
    }
}
