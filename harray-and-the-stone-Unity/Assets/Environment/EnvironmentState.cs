using System.Collections;
using System.Collections.Generic;

public class GameMessage
{
    public EnvironmentState state = new EnvironmentState();
    public string lastReward;
    public bool isEnd;
}
public class EnvironmentState
{
    public string harryX , harryY;
    public string theiveX, theiveY;
}
public class EnviornmentAction
{
    public bool isEnd;
    public string harryCommand = "";
}
