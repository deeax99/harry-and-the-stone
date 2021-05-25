using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;
public class deleteME : MonoBehaviour
{
    [SerializeField] float lerpSpeed = 140;
    private void Update()
    {
        float x = Input.GetAxis("Horizontal");
        float y = Input.GetAxis("Vertical");

        float angle = Mathf.Atan2(y, x) * Mathf.Rad2Deg;
        angle = angle;

        transform.rotation = Quaternion.Lerp(transform.rotation, Quaternion.Euler(0, 0, angle), lerpSpeed * Time.deltaTime);
    }

}
