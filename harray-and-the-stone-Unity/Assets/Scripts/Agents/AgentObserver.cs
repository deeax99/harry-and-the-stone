using UnityEngine;

public class AgentObserver
{
    private Transform observer;
    private VisionTest visionTest;
    private int startIndex;
    private bool updateFrame = false;
    public AgentObserver(int startIndex, Transform observer, VisionTest visionTest) : this(startIndex  ,observer)
    {
        updateFrame = true;
        this.visionTest = visionTest;
    }
    public AgentObserver(int startIndex, Transform observer)
    {
        this.startIndex = startIndex;
        this.observer = observer;
    }
    public void Initialization(object[] agnetState, Vector2 position)
    {
        UpdateValues(agnetState, position, 1);
    }
    public void Update(object[] agnetState , int frame)
    {
        if (IsDead())
        {
            UpdateValues(agnetState, Vector2.zero, 0);
        }
        else if (VisionTest())
        {
            UpdateValues(agnetState, observer.position, frame);
        }
    }
    void UpdateValues(object[] agnetState, Vector2 position , int frame)
    {
        agnetState[startIndex] = position.x / 5;
        agnetState[startIndex + 1] = position.y / 5;
        if (updateFrame)
        {
            agnetState[startIndex + 2] = frame / 200f;
        }
    }
    public virtual bool IsDead()
    {
        return false;
    }
    public virtual bool VisionTest()
    {
        if (visionTest == null)
            return true;
        visionTest.UpdateLine();
        return visionTest.isVisiable;
    }
}
public class ThieveObserver : AgentObserver
{
    private Thieve thieve;
    public ThieveObserver(int startIndex, Transform observer, VisionTest visionTest, Thieve thieve) : base(startIndex, observer, visionTest)
    {
        this.thieve = thieve;
    }
    public ThieveObserver(int startIndex, Transform observer, Thieve thieve) : base(startIndex, observer)
    {
        this.thieve = thieve;
    }
    public override bool IsDead()
    {
        return thieve.isDead;
    }
}