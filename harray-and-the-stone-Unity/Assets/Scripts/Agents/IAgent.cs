using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface IAgent
{
    void AgentStart();
    void ApplyAction(EnviornmentAction action, int frame);
    void UpdateState(int frame);
    void ResetState();
}
