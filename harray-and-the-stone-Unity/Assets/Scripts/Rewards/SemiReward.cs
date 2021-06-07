using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SemiReward : MonoBehaviour
{
    [SerializeField] private int agentID;
    private void OnTriggerEnter2D(Collider2D collision)
    {
        if (AgentsManager.instance.AgentCheck(collision.gameObject, agentID))
        {
            AgentsManager.instance.SemiRewardTrigger(GetInstanceID(), agentID);
        }
    }
}
