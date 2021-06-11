using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameLoop : MonoBehaviour
{
    
    [SerializeField] private AgentsManager agentsManager;

    private int frame = 1;
    public void InitLoop()
    {
        agentsManager.AgentStart();
    }
    public EnvironmentState UpdateLoop(EnviornmentAction action)
    {
        if (action.done)
        {
            frame = 1;
            agentsManager.ResetGame();
        }
        else
        {
            frame++;
            agentsManager.ApplyAction(action, frame);
        }
        Physics2D.Simulate(1 / 60f);
        return GetEnviromantState();
    }
    EnvironmentState GetEnviromantState()
    {
        return agentsManager.GetEnviromantState(frame);
    }
}
