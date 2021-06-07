using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Thieve : MonoBehaviour, IAgent
{
    const int THIEVE_STATE_SIZE = 14;

    const float speed = .075f;
    const float diamondSpeed = .04f;

    const float lerpSpeed = 15;
    [SerializeField] public int thieveID;

    [SerializeField] private Transform harryTransform;
    [SerializeField] private Transform theOtherThieveTransform;
    [SerializeField] private Transform firstDiamondTransform, secondDiamondTransform;

    [SerializeField] private Harry harry;
    [SerializeField] private VisionTest harryVision;

    [SerializeField] private Thieve otherThieve;
    [SerializeField] private Diamond firstDiamond, secondDiamond;


    private AgentObserver[] agentObservers;

    [HideInInspector] public Vector3 initialPosition;
    [HideInInspector] public int carryStatus = 0;
    [HideInInspector] public bool isDead = false;


    public void AgentStart()
    {
        InitializeState();
        InitializeVision();
        InitializeObserver();
        InitializePosition();
    }
    void InitializeState()
    {
        var state = AgentsManager.instance.environmentState;
        if (thieveID == 1)
        {
            state.firstThieve = new object[THIEVE_STATE_SIZE];
        }
        else if (thieveID == 2)
        {
            state.secondThieve = new object[THIEVE_STATE_SIZE];
        }
        else
        {
            Debug.LogError("Wrong thieve ID");
        }
    }
    void InitializeVision()
    {
        harryVision.InitializationVisionTest(transform, harryTransform);
    }
    void InitializeObserver()
    {
        agentObservers = new AgentObserver[5];

        agentObservers[0] = new AgentObserver(0, transform); //thieve me [0 , 1]
        agentObservers[1] = new ThieveObserver(2, theOtherThieveTransform, otherThieve); //thieve other [2 , 3]

        agentObservers[2] = new AgentObserver(4, harryTransform, harryVision); //harry [4 , 6]

        agentObservers[3] = new AgentObserver(7, firstDiamondTransform); // diamond 1 [7 , 8]
        agentObservers[4] = new AgentObserver(9, firstDiamondTransform); // diamond 1 [9 , 10]

    }
    void InitializePosition()
    {
        var thieveState = GetThieveState();

        initialPosition = transform.position;

        agentObservers[0].Initialization(thieveState, initialPosition);

        agentObservers[1].Initialization(thieveState, otherThieve.initialPosition);
        agentObservers[2].Initialization(thieveState, harry.initialPosition);

        agentObservers[3].Initialization(thieveState, firstDiamondTransform.position);
        agentObservers[4].Initialization(thieveState, secondDiamondTransform.position);

    }

    public void ApplyAction(EnviornmentAction action, int frame)
    {
        if (isDead)
            return;

        int grabDiamond = thieveID == 1 ? action.firstThieveGrab : action.secondThieveGrab;
        Vector2 thievePosition = Vector2.zero;

        if (thieveID == 1)
        {
            thievePosition = new Vector2(action.firstThieveX, action.firstThieveY);
        }
        else if (thieveID == 2)
        {
            thievePosition = new Vector2(action.secondThieveX, action.secondThieveY);
        }
        thievePosition.Normalize();

        if (thievePosition.sqrMagnitude > .1f * .1f)
        {
            float angle = Mathf.Atan2(thievePosition.y, thievePosition.x) * Mathf.Rad2Deg;
            transform.rotation = Quaternion.Lerp(transform.rotation, Quaternion.Euler(0, 0, angle), lerpSpeed);
        }
        if (grabDiamond == 0)
        {
            firstDiamond.DropDiamond(this);
            secondDiamond.DropDiamond(this);
        }
        else
        {
            firstDiamond.GrabDiamond(this);
            secondDiamond.GrabDiamond(this);
        }
        float thieveSpeed = carryStatus == 0 ? speed : diamondSpeed;
        transform.position += (Vector3)thievePosition * thieveSpeed;
    }

    public void UpdateState(int frame)
    {
        if (isDead)
            return;

        var thieveState = GetThieveState();

        foreach (AgentObserver observer in agentObservers)
        {
            observer.Update(thieveState, frame);
        }

        thieveState[11] = carryStatus;
        thieveState[12] = otherThieve.carryStatus;
        thieveState[13] = frame / 200f;

    }

    public void ResetState()
    {
        transform.position = initialPosition;
        carryStatus = 0;
        isDead = false;
        InitializeState();
        InitializePosition();

    }
    public void Die()
    {
        isDead = true;

        firstDiamond.DropDiamond(this);
        secondDiamond.DropDiamond(this);

        if (thieveID == 1)
        {
            AgentsManager.instance.environmentState.firstThieveEnd = true;
        }
        else if (thieveID == 2)
        {
            AgentsManager.instance.environmentState.secondThieveEnd = true;
        }

        transform.position = Vector2.one * 10000;
        DestoryState();
    }
    object[] GetThieveState()
    {
        var state = AgentsManager.instance.environmentState;
        if (thieveID == 1)
        {
            return state.firstThieve;
        }
        else if (thieveID == 2)
        {
            return state.secondThieve;
        }
        else
        {
            throw new System.Exception("Wrong thieveID");
        }
    }
    void DestoryState()
    {
        var state = AgentsManager.instance.environmentState;
        if (thieveID == 1)
        {
            state.firstThieve = new object[0];
        }
        else if (thieveID == 2)
        {
            state.secondThieve = new object[0];
        }
    }

}
