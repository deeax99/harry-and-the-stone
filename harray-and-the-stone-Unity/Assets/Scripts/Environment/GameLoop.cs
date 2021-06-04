using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameLoop : MonoBehaviour
{

    private int frame = 1;
    private void Start()
    {
        AgentsManager.instance.AgentStart();
    }
    private void Update()
    {
        GetAndApplyAction();
        SendAction();
    }
    void GetAndApplyAction()
    {
        var action = MLCommunication.GetAction();
        if (action.done)
        {
            frame = 1;
            AgentsManager.instance.ResetGame();
        }
        else
        {
            frame++;
            AgentsManager.instance.ApplyAction(action, frame);
        }
        Physics2D.Simulate(1 / 60f);
    }
    void SendAction()
    {
        MLCommunication.SendAction(GetEnviromantState());
    }
    EnvironmentState GetEnviromantState()
    {
        return AgentsManager.instance.GetEnviromantState(frame);
    }
}
