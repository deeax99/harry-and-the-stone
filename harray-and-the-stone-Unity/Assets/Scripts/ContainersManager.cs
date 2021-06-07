using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ContainersManager : MonoBehaviour
{
    const int PORT = 7979;
    const int COUNT = 50;
    const float offsetSize = 12;

    [SerializeField] private GameObject containerPrefab;

    private GameLoop[] containers;
    private MLCommunication mlCommunication;

    public int GetPort()
    {
        string[] args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--p")
            {
                return int.Parse(args[i + 1]);
            }
        }
        return PORT;
    }
    public int ContainersCount()
    {
        string[] args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--c")
            {
                return int.Parse(args[i + 1]);
            }
        }
        return COUNT;
    }
    private void Start()
    {
        int count = ContainersCount();
        int port = GetPort();

        int grid = Mathf.CeilToInt(Mathf.Sqrt(count));

        mlCommunication = new MLCommunication(port);
        containers = new GameLoop[count];

        for (int i = 0; i < count; i++)
        {
            var container = Instantiate(containerPrefab, transform);

            container.transform.localPosition = new Vector2(i % grid, i / grid) * offsetSize;
            
            containers[i] = container.GetComponent<GameLoop>();
            containers[i].InitLoop();
        }
    }
    private void Update()
    {
        var action = mlCommunication.GetAction();

        for (int i = 0; i < containers.Length; i++)
            if (i != action.continerID)
                containers[action.continerID].gameObject.SetActive(false);
        var state = containers[action.continerID].UpdateLoop(action);

        for (int i = 0; i < containers.Length; i++)
            containers[action.continerID].gameObject.SetActive(true);

        mlCommunication.SendAction(state);
    }
}
