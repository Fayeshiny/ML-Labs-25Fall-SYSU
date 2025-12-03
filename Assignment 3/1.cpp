binary_semaphore mutex = 1;        // 保护计数器
binary_semaphore sem[2] = {1, 1};  // sem[0] 作为桥的互斥量
int count[2] = {0, 0};             // 当前在桥上的左右人数

void P(bool isRight, int id)
{
    int d = isRight ? 1 : 0;       // 0=左，1=右

    while (true) {
        /* 进入区：准备上桥 */
        P(mutex);
        count[d]++;                // 本方向人数 +1
        if (count[d] == 1)         // 本方向第一个人上桥
            P(sem[0]);             // 占有这座桥，禁止反向
        V(mutex);

        /* ---- 临界区：过桥 ---- */
        // cross_bridge(d, id);

        /* 退出区：下桥 */
        P(mutex);
        count[d]--;                // 本方向人数 -1
        if (count[d] == 0)         // 本方向最后一个人离开
            V(sem[0]);             // 释放桥，允许对向上桥
        V(mutex);

        /* 余下非临界部分 */
    }
}