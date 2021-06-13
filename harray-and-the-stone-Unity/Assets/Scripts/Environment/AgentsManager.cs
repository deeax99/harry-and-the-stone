using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public class AgentsManager : MonoBehaviour
{
    public const float PERCESISION = .1f;
    public const int MAX_FRAME = 350;

    [SerializeField] private MonoBehaviour[] agents;

    [SerializeField] private Transform harry;
    [SerializeField] private Transform firstThieve, secondThieve;
    [SerializeField] private Transform firstDiamond, secondDiamond;

    [SerializeField] private float diamondThreshold = -2;

    private HashSet<(int, int)> usedInstance = new HashSet<(int, int)>();
    private bool[] takeDimond = new bool[2];

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

    public void ResetGame()
    {
        environmentState = new EnvironmentState();
        foreach (IAgent agent in agents)
        {
            agent.ResetState();
        }
        usedInstance = new HashSet<(int, int)>();
        takeDimond = new bool[2];
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

    private int firstThieveAdd = 0;
    private int secondThieveAdd = 0;
    private int harryAdd = 0;

    void UpdateReward(int frame)
    {
        var state = environmentState;

        state.harryReward = harryAdd;
        state.firstThievesReward = firstThieveAdd;
        state.secondThieveReward = secondThieveAdd;

        DiamondTaked();

        if ((state.firstThieveEnd && state.secondThieveEnd) || frame == MAX_FRAME)
        {
            state.done = true;
            state.harryReward = 50;
            state.firstThievesReward = -50;
            state.secondThieveReward = -50;
        }
        else if (IsThieveWinning())
        {
            state.done = true;
            state.harryReward = -50;
            state.firstThievesReward = 50;
            state.secondThieveReward = 50;
        }

        firstThieveAdd = 0;
        secondThieveAdd = 0;
        harryAdd = 0;

        state.harryReward /= 10f;
        state.firstThievesReward /= 10f;
        state.secondThieveReward /= 10f;
    }
    bool IsThieveWinning()
    {
        return firstDiamond.localPosition.y < diamondThreshold || secondDiamond.localPosition.y < diamondThreshold;
    }
    public bool AgentCheck(GameObject agent, int agentID)
    {
        if (agentID == 0)
        {
            return agent == harry.gameObject;
        }
        else if (agentID == 1)
        {
            return agent == firstThieve.gameObject;
        }
        else
        {
            return agent == secondThieve.gameObject;
        }
    }
    public void DiamondTaked()
    {
        Transform[] thieves = new Transform[] { firstThieve,secondThieve };
        for (int i = 0; i < 2; i++)
            if (!takeDimond[i] && thieves[i].GetComponent<Thieve>().carryStatus == i + 1)
            {
                print($"Agnet {i}");
                var state = environmentState;
                takeDimond[i] = true;
                if (i == 0)
                {
                    state.firstThievesReward += 7;
                    usedInstance.RemoveWhere(((int, int) key) => key.Item2 == 1);
                }
                else
                {
                    state.secondThieveReward += 7;
                    usedInstance.RemoveWhere(((int, int) key) => key.Item2 == 2);
                }
            }
    }
    public void SemiRewardTrigger(int instanceID, int agentID)
    {
        if (usedInstance.Add((instanceID, agentID)))
        {
            if (agentID == 0)
            {
                harryAdd += 10;
            }
            else if (agentID == 1)
            {
                firstThieveAdd += 3;
            }
            else
            {
                secondThieveAdd += 3;
            }
        }
    }
}
