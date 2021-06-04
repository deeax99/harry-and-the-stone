using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public class AgentsManager : MonoBehaviour
{
    public const float PERCESISION = .1f;
    public const int MAX_FRAME = 2550;

    public static AgentsManager instance;

    [SerializeField] private MonoBehaviour[] agents;
    [SerializeField] private Transform firstDiamond, secondDiamond;
    [SerializeField] private float diamondThreshold = -2;

    private void OnValidate()
    {
        if (agents == null) return;
        int agentsLength = agents.Length;
        for (int i = 0; i < agentsLength; i++)
        {
            if (agents[i] is IAgent == false)
            {
                agents[i] = null;
            }
        }
    }


    public EnvironmentState environmentState;

    int frameCount = 0;

    private void Awake()
    {
        instance = this;
    }

    public void ResetGame()
    {
        environmentState = new EnvironmentState();
        foreach (IAgent agent in agents)
        {
            agent.ResetState();
        }
    }

    public EnvironmentState GetEnviromantState(int frame)
    {
        foreach (IAgent agent in agents)
        {
            agent.UpdateState(frame);
        }
        return environmentState;
    }

    public void ApplyAction(EnviornmentAction action, int frame)
    {
        foreach (IAgent agent in agents)
        {
            agent.ApplyAction(action, frame);
        }
        UpdateReward(frame);
    }
    public void AgentStart()
    {
        environmentState = new EnvironmentState();
        foreach (IAgent agent in agents)
        {
            agent.AgentStart();
        }
    }
    void UpdateReward(int frame)
    {
        var state = environmentState;
        if ((state.firstThieveEnd && state.secondThieveEnd) || frame == MAX_FRAME)
        {
            state.done = true;
            state.harryReward = 1;
            state.thievesReward = -1;
        }
        else if (isThieveWinning())
        {
            state.done = true;
            state.harryReward = -1;
            state.thievesReward = 1;
        }
    }
    bool isThieveWinning()
    {
        return firstDiamond.localPosition.y < diamondThreshold || secondDiamond.localPosition.y < diamondThreshold;
    }
}
