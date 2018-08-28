package com.example.anupam.test2;

public class LeaveForm {
    int ndays;
    String approval;
    String address;
    String regdate;
    LeaveForm()
    {

    }

    public LeaveForm(int ndays, String approval, String address, String regdate) {
        this.ndays = ndays;
        this.approval = approval;
        this.address = address;
        this.regdate = regdate;
    }

    public int getNdays() {
        return ndays;
    }

    public void setNdays(int ndays) {
        this.ndays = ndays;
    }

    public String getApproval() {
        return approval;
    }

    public void setApproval(String approval) {
        this.approval = approval;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getRegdate() {
        return regdate;
    }

    public void setRegdate(String regdate) {
        this.regdate = regdate;
    }
}
