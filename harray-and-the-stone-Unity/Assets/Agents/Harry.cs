using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Harry : MonoBehaviour , IAgent
{
    [SerializeField] private float jumpSize = .1f;
    [SerializeField] private Transform theive;

    void Start()
    {
        EnvironmentElements.instance.agents.Add(this);
    }

    public void ApplyAction(EnviornmentAction action)
    {
        switch (action.harryCommand)
        {
            case "left":
                transform.position += Vector3.left * jumpSize;
                break;
            case "right":
                transform.position += Vector3.right * jumpSize;
                break;
            case "up":
                transform.position += Vector3.up * jumpSize;
                break;
            case "down":
                transform.position += Vector3.down * jumpSize;
                break;
        }

        EnvironmentElements.instance.environmentState.harryX = Mathf.RoundToInt(transform.position.x / jumpSize).ToString();
        EnvironmentElements.instance.environmentState.harryY = Mathf.RoundToInt(transform.position.y / jumpSize).ToString();

        UpdateReward();
    }
    void UpdateReward()
    {
        int reward;
        if (Mathf.Abs(transform.position.x) > 6 || Mathf.Abs(transform.position.y) > 4.5f)
        {
            reward = -100;
            transform.position = Vector2.zero;
        }
        else
        {
            if (Vector2.Distance(theive.position, transform.position) > 1.5f)
                reward = 1;
            else 
                reward = -1;
        }
        EnvironmentElements.instance.environmentState.lastReward = reward.ToString();
    }
}
