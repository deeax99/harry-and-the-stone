using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Harry : MonoBehaviour , IAgent
{
    const int HARRY_STATE_SIZE = 15;

    const float speed = .05f;
    const float lerpSpeed = 15;

    [SerializeField] private Transform firstThieveTransform, secondThieveTransform;
    [SerializeField] private Transform firstDiamondTransform , secondDiamondTransform;

    [SerializeField] private VisionTest firstThieveVision , secondThieveVision;
    [SerializeField] private VisionTest firstDiamondVision, secondDiamondVision;

    [SerializeField] private Thieve firstThieve, secondThieve;

    [SerializeField] private Diamond firstDiamond, secondDiamond;

    [HideInInspector] public Vector3 initialPosition;
    private AgentObserver[] agentObservers;


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
        state.harryState = new object[HARRY_STATE_SIZE];
    }

    void InitializeVision()
    {
        firstThieveVision.InitializationVisionTest(transform, firstThieveTransform);
        secondThieveVision.InitializationVisionTest(transform, secondThieveTransform);
        firstDiamondVision.InitializationVisionTest(transform, firstDiamondTransform);
        secondDiamondVision.InitializationVisionTest(transform, secondDiamondTransform);
    }
    void InitializeObserver()
    {
        agentObservers = new AgentObserver[5];

        agentObservers[0] = new AgentObserver(0, transform); //harry [0 , 1]
        
        agentObservers[1] = new ThieveObserver(2, firstThieveTransform, firstThieveVision, firstThieve); //thieve 1 [2 , 4]
        agentObservers[2] = new ThieveObserver(5, secondThieveTransform, secondThieveVision, secondThieve); //thieve 2 [5 , 7]

        agentObservers[3] = new AgentObserver(8, firstDiamondTransform , firstDiamondVision); // diamond 1 [8 , 10]
        agentObservers[4] = new AgentObserver(11, firstDiamondTransform, firstDiamondVision); // diamond 1 [11 , 13]

    }
    void InitializePosition()
    {
        var harryState = AgentsManager.instance.environmentState.harryState;
        initialPosition = transform.position;
        
        agentObservers[0].Initialization(harryState, initialPosition);
        
        agentObservers[1].Initialization(harryState, firstThieve.initialPosition);
        agentObservers[2].Initialization(harryState, secondThieve.initialPosition);
        
        agentObservers[3].Initialization(harryState, firstDiamond.initialPosition);
        agentObservers[4].Initialization(harryState, secondDiamond.initialPosition);
    }

    public void ApplyAction(EnviornmentAction action, int frame)
    {
        Vector2 harryPosition = Vector2.zero;
        harryPosition.Normalize();

        if (harryPosition.sqrMagnitude > .1f * .1f)
        {
            float angle = Mathf.Atan2(harryPosition.y, harryPosition.x) * Mathf.Rad2Deg;
            transform.rotation = Quaternion.Lerp(transform.rotation, Quaternion.Euler(0, 0, angle), lerpSpeed);
        }
        transform.position += (Vector3)harryPosition * speed;
    }
    public void UpdateState(int frame)
    {
        var harryState = AgentsManager.instance.environmentState.harryState;

        foreach (AgentObserver observer in agentObservers)
        {
            observer.Update(harryState, frame);
        }

        harryState[14] = frame;

    }
    public void ResetState()
    {
        transform.position = initialPosition;
        InitializeState();
        InitializePosition();
    }

    public void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.tag == "thieve")
        {
            var thieve = collision.gameObject.GetComponent<Thieve>();
            thieve.Die();
        }
    }
}
