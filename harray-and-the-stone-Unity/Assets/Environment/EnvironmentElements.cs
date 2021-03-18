using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentElements : MonoBehaviour
{
    public static EnvironmentElements instance;

    public List<IAgent> agents;
    public EnvironmentState environmentState;

    private void Awake()
    {
        agents = new List<IAgent>();
        environmentState = new EnvironmentState();
        instance = this;
    }

    public EnvironmentState GetState()
    {
        return environmentState;
    }

    public void ApplyAction(EnviornmentAction action)
    {
        foreach (IAgent agent in agents)
        {
            agent.ApplyAction(action);
        }
    }
}
