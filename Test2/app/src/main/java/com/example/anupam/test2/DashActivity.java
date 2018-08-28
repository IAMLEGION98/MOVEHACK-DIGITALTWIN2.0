package com.example.anupam.test2;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.location.Address;
import android.location.Criteria;
import android.location.Geocoder;
import android.location.Location;
import android.location.LocationManager;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.List;


public class DashActivity extends AppCompatActivity {

    TextView applyleave,back,status,history;
    private static final int LOCATION_REQUEST = 500;
    DatabaseReference statusbase;
    double latitude;
    String gencity=null;
    double longitude;
    String eid=null;
    int flag=0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_dash);
        runcourse();
        Log.e("The Latitude is ",String.valueOf(latitude));
        Log.e("The Longitude is ",String.valueOf(longitude));
        getAddress(latitude,longitude);


        statusbase = FirebaseDatabase.getInstance().getReference("empleave");
        status = (TextView)findViewById(R.id.tvstatus);
        history =(TextView)findViewById(R.id.tvprev);
        back = (TextView)findViewById(R.id.etback);
        applyleave = (TextView) findViewById(R.id.tvgoapply);
        Intent intent1 = getIntent();

        eid = intent1.getStringExtra("employeeid");
        applyleave.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                statusbase.addValueEventListener(new ValueEventListener() {
                                                     @Override
                                                     public void onDataChange(DataSnapshot dataSnapshot) {

                                                         for (DataSnapshot postsnapshot : dataSnapshot.getChildren()) {
                                                             String idval = postsnapshot.getKey().toString();
                                                             if (idval.equals(eid)) {
                                                                 flag = 1;
                                                                 Toast.makeText(getApplicationContext(), "You have pending leave approval ", Toast.LENGTH_SHORT).show();
                                                             }
                                                         }
                                                         if(flag==0) {
                                                             Intent intent = new Intent(getApplicationContext(), LeaveActivity.class);
                                                             intent.putExtra("employeeid", eid);
                                                             intent.putExtra("address",gencity);
                                                             startActivity(intent);
                                                         }

                                                     }

                                                     @Override
                                                     public void onCancelled(DatabaseError databaseError) {

                                                     }
                                                 });

            }
        });

        status.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
             Intent intents = new Intent(getApplicationContext(),LeaveStatus.class);
             intents.putExtra("employeeid",eid);
             startActivity(intents);
            }
        });

    }

    public List<Address> getAddress(double latitude, double longitude) {
        List<Address> addresses = null;
        try {

            Geocoder geocoder;

            geocoder = new Geocoder(DashActivity.this);
            if (latitude != 0 || longitude != 0) {
                addresses = geocoder.getFromLocation(latitude, longitude, 1);

                //testing address below

                String address = addresses.get(0).getAddressLine(0);
                String city = addresses.get(0).getAddressLine(1);
                String country = addresses.get(0).getAddressLine(2);
                gencity= "Address: "+address;
                Toast.makeText(getApplicationContext(),"You are at "+"address = " + address + ", city =" + city
                        + ", country = " + country,Toast.LENGTH_LONG).show();

            }

        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            // Toast.makeText(this, e.getMessage(), Toast.LENGTH_SHORT).show();
        }
        return addresses;
    }
    public void runcourse()
    {
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,new String[]{ android.Manifest.permission.ACCESS_FINE_LOCATION}, LOCATION_REQUEST);
            return;
        }
        LocationManager locationmanager = (LocationManager) getSystemService(LOCATION_SERVICE);
        Criteria cr = new Criteria();
        String provider = locationmanager.getBestProvider(cr, true);
        Location location = locationmanager.getLastKnownLocation(provider);
        if (location != null) {
            latitude=location.getLatitude();
            longitude=location.getLongitude();
        }

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode)
        {
            case LOCATION_REQUEST:
                if(grantResults.length>0 && grantResults[0]== PackageManager.PERMISSION_GRANTED )
                {
                   runcourse();
                }
                break;
        }
    }
}
