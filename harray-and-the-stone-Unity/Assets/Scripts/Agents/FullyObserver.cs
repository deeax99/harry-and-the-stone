using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FullyObserver : MonoBehaviour, IAgent
{
    const int FULLY_STATE_SIZE = 13;

    [SerializeField] private Harry harry;
    [SerializeField] private Thieve firstThieve, secondThieve;
    [SerializeField] private Diamond firstDiamond, secondDiamond;

    private AgentObserver[] agentObservers;

    public void AgentStart()
    {
        InitializeState();
        InitializeObserver();
        InitializePosition();
    }

    private void InitializeState()
    {
        var state = AgentsManager.instance.environmentState;
        state.fullState = new object[FULLY_STATE_SIZE];
    }

    void InitializeObserver()
    {
        agentObservers = new AgentObserver[5];

        agentObservers[0] = new AgentObserver(0, harry.transform); //harry [0 , 1]

        agentObservers[1] = new AgentObserver(2, firstThieve.transform); //thieve 1 [2 ,3]
        agentObservers[2] = new AgentObserver(4, secondThieve.transform); //thieve 2 [4 , 5]

        agentObservers[3] = new AgentObserver(6, firstDiamond.transform); // diamond 1 [6 , 7]
        agentObservers[4] = new AgentObserver(8, secondDiamond.transform); // diamond 1 [8 , 9]

    }
    void InitializePosition()
    {
        var fullyState = AgentsManager.instance.environmentState.fullState;

        agentObservers[0].Initialization(fullyState, harry.initialPosition);

        agentObservers[1].Initialization(fullyState, firstThieve.initialPosition);
        agentObservers[2].Initialization(fullyState, secondThieve.initialPosition);

        agentObservers[3].Initialization(fullyState, firstDiamond.initialPosition);
        agentObservers[4].Initialization(fullyState, secondDiamond.initialPosition);
    }


    public void ApplyAction(EnviornmentAction action, int frame) { }

    public void ResetState()
    {
        InitializeState();
        InitializePosition();
    }

    public void UpdateState(int frame)
    {
        var fullyState = AgentsManager.instance.environmentState.fullState;

        foreach (AgentObserver observer in agentObservers)
        {
            observer.Update(fullyState, frame);
        }

        fullyState[10] = firstThieve.carryStatus;
        fullyState[11] = secondThieve.carryStatus;

        fullyState[12] = frame;
    }

}
