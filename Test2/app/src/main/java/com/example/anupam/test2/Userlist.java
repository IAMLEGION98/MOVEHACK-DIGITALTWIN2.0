package com.example.anupam.test2;

import android.app.Activity;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import java.util.Iterator;
import java.util.List;

public class Userlist extends ArrayAdapter<LeaveForm> {

    private Activity context;
    private List<LeaveForm> userlist;
    private List userids;
   

    public Userlist(Activity context, List<LeaveForm> userslist,List useridlist)
    {
        super(context,R.layout.list_layout,userslist);
        this.context = context;
        this.userlist = userslist;
        this.userids= useridlist;

    }

    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        LayoutInflater inflater = context.getLayoutInflater();
        View listviewitem = inflater.inflate(R.layout.list_layout,null, true);
        TextView tvmid = (TextView)listviewitem.findViewById(R.id.tvmid);
        TextView tvmdate = (TextView)listviewitem.findViewById(R.id.tvmdate);
        TextView tvmapp = (TextView)listviewitem.findViewById(R.id.tvmapp);
        TextView tvmloc = (TextView)listviewitem.findViewById(R.id.tvmlocation);
        LeaveForm lform = userlist.get(position);
        tvmid.setText(String.valueOf(userids.get(position)));
        tvmdate.setText(lform.getRegdate());
        tvmapp.setText(lform.getApproval());
        tvmloc.setText(lform.getAddress());
        return listviewitem;
    }
}
