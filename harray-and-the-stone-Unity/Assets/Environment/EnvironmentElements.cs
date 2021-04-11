using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentElements : MonoBehaviour
{
    public const float PERCESISION = .1f;
    public const int MAX_FRAME = 2550;

    public static EnvironmentElements instance;

    public List<IAgent> agents;
    public EnvironmentState environmentState
    {
        get { return gameMessage.state; }
    }
    private GameMessage gameMessage;

    [SerializeField] private Transform harry , theive;

    int frameCount = 0;

    private void Awake()
    {
        agents = new List<IAgent>();
        gameMessage = new GameMessage();
        instance = this;
    }

    public void ResetGame()
    {
        gameMessage = new GameMessage();
        foreach (IAgent agent in agents)
        {
            agent.ResetState();
        }
    }

    public GameMessage GetGameMessage()
    {
        foreach (IAgent agent in agents)
        {
            agent.UpdateState();
        }
        UpdateReward();
        return gameMessage;
    }

    public void ApplyAction(EnviornmentAction action)
    {
        foreach (IAgent agent in agents)
        {
            agent.ApplyAction(action);
        }
    }

    void UpdateReward()
    {
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
    }


}
