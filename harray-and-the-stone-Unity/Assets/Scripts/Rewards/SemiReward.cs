using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SemiReward : MonoBehaviour
{
    [SerializeField] private int agentID;
    [SerializeField] private AgentsManager agentsManager;
    private void OnTriggerEnter2D(Collider2D collision)
    {
        if (agentsManager.AgentCheck(collision.gameObject, agentID))
        {
            agentsManager.SemiRewardTrigger(GetInstanceID(), agentID);
        }
    }
}
