using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public class AgentsManager : MonoBehaviour
{
    public const float PERCESISION = .1f;
    public const int MAX_FRAME = 2550;

    public static AgentsManager instance;

    [SerializeField] private MonoBehaviour[] agents;

    private void OnValidate()
    {
        if (agents == null) return;
        int agentsLength = agents.Length;
        for (int i =0; i < agentsLength; i++)
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
        UpdateReward();
        return environmentState;
    }

    public void ApplyAction(EnviornmentAction action , int frame)
    {
        foreach (IAgent agent in agents)
        {
            agent.ApplyAction(action , frame);
        }
    }
    public void AgentStart()
    {
        environmentState = new EnvironmentState();
        foreach (IAgent agent in agents)
        {
            agent.AgentStart();
        }
    }
    void UpdateReward()
    {
        /*
        int reward = 0;
        if (Mathf.Abs(harry.position.x) > 6 || Mathf.Abs(harry.position.y) > 4.5f)
        {
            //reward = -1500;
            frameCount = 0;
            gameMessage.isEnd = true;
        }
        else
        {
            if (Vector2.Distance(theive.position, harry.position) < .5f)
            {
                //reward = -1500;
                frameCount = 0;
                print(harry.position);
                gameMessage.isEnd = true;
            }
            else
            {
                reward = 1;
            }
        }

        if (++frameCount == MAX_FRAME)
        {
            gameMessage.isEnd = true;
            if (reward == 1)
            {
                reward = 20;
            }
            frameCount = 0;
        }
        gameMessage.lastReward = reward.ToString();
        */
    }
}
