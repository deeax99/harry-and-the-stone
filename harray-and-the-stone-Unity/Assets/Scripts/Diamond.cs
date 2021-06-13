using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Diamond : MonoBehaviour , IAgent
{
    [SerializeField] private int diamondID;

    [SerializeField] private Thieve[] thieves;

    [HideInInspector] public int followThieve;
    [HideInInspector] public Vector3 initialPosition;

    public void AgentStart()
    {
        initialPosition = transform.localPosition;
    }

    public void ApplyAction(EnviornmentAction action, int frame)
    {
        if (followThieve > 0)
        {
            transform.localPosition = thieves[followThieve - 1].transform.localPosition;
        }
    }

    public void UpdateState(int frame)
    {
        
    }

    public void ResetState()
    {
        followThieve = 0;
        transform.localPosition = initialPosition;
    }

    public void GrabDiamond(Thieve thieve)
    {
        if (thieve.carryStatus == 0 && CanGrab(thieve.transform.localPosition))
        {
            thieve.carryStatus = thieve.thieveID;
            followThieve = thieve.thieveID;
        }
    }
    public void DropDiamond(Thieve thieve)
    {
        if (thieve.thieveID == followThieve)
        {
            thieve.carryStatus = 0;
            followThieve = 0;
        }
    }
    bool CanGrab(Vector2 position)
    {
        const float MAX_DISTANCE = 1;
        return Vector2.Distance(transform.localPosition, position) < MAX_DISTANCE;
    }
}
