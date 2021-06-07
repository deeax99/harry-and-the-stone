using System.Collections;
using System.Collections.Generic;
/*
public class GameMessage
{
    public EnvironmentState state = new EnvironmentState(2);
    public string lastReward;
    public bool isEnd;
}
*/
public class EnvironmentState
{
    public bool done;
    public bool firstThieveEnd, secondThieveEnd;
    public float harryReward, firstThievesReward , secondThieveReward;

    public object[] harryState;
    public object[] firstThieve;
    public object[] secondThieve;
    public object[] fullState;

}
public class EnviornmentAction
{
    public bool done;
    public float harryX, harryY;
    public float firstThieveX, firstThieveY;
    public float secondThieveX, secondThieveY;
    public int firstThieveGrab , secondThieveGrab;
}
/*
public class FullState 13 input
{
    public float harryX, harryY;
    public float thieve1X, thieve1Y;
    public float thieve2X, thieve2Y;
    public float diamond1X, diamond1Y;
    public float diamond2X, diamond2Y;
    public int thieve1Carry, thieve2Carry;
    public int frame;
}
public class HarryState //15 input
{
    public float harryX, harryY;
    
    public float thieve1X, thieve1Y;
    public float thieve2X, thieve2Y;
    
    public float diamond1X, diamond1Y;
    public float diamond2X, diamond2Y;

    public int frame;
    public int lastDiamond1, lastDiamond2;
    public int lastThieve1, lastThieve2; //0 for death

}
public class ThieveState 14 input
{
    public float thieveX, thieveY;

    public float harryX, harryY;
    public float otherThieveX, otherThieveY;

    public float diamond1X, diamond1Y;
    public float diamond2X, diamond2Y;
    
    public int carry;
    other carry

    public int frame;
    public int lastHarry;
}
*/