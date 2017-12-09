#include <map>
#include <vector>
#include <algorithm>
#include <iostream>
#include "cfunc.h"

int get_class_c(int count, std::map<int, int> hand) {

    int Invalid = 0; //未知
    int Single = 100; //单张
    int Pair = 200; //对子
    int Triplet = 300; //三条
    int Straight = 400; //单顺
    int ThreePairs = 500; //双顺
    int TwoTriplet = 600; //双顺
    // int ThreeSeq = ; //三顺
    // int ThreePlus = ; //三带一（一张或一对）
    // int Airplane = ; //飞机
    // int FourSeq = ; //四带二（两张或两对）
    int Bomb = 700; //炸弹
    int Nuke = 799;//王炸
    int NumMax = 0; //同牌面的最大数量
    int ValueMax = 0; //#最大数量的最大权值

    //判断是否为王炸
    if (count == 2 &&
        hand.find(16) != hand.end() &&
        hand.find(17) != hand.end()){
        return Nuke;
    }
    //找出相同牌面的最大数量，和最大权值
    for (auto mem : hand){
        if (mem.second >= NumMax && mem.first > ValueMax){
            NumMax = mem.second;
            ValueMax = mem.first;
        }
    }
    //根据牌面相同的最大数量判断类型
    switch (NumMax){
    case 1:
        if (count == 1){//单张
            return Single + ValueMax;
        }
        else if (count >= 5){//判断是否为顺子
            int begin = 0;
            for (auto mem : hand){
                if (!begin)
                    begin = mem.first;
                if (begin++ != mem.first || mem.first >= 15)//牌不是连续的或者带了2及以上的牌
                    return Invalid;
            }
            return Straight + ValueMax; //单顺
        }
        return Invalid;
    case 2:
        if (count == 2){//一对
            return Pair + ValueMax;
        }
        if (count == 6 && !(count % 2)){//连对
            int begin = 0;
            for (auto mem : hand){//确定牌是连续的，并且都是成对的
                if (!begin)
                    begin = mem.first;
                if (begin++ != mem.first || mem.second != 2)//牌不符合规定
                    return Invalid;
            }
            return ThreePairs + ValueMax;
        }
        return Invalid;//牌不符合规定
    case 3:
    {
        if (count == 3){//三条
            return Triplet + ValueMax;
        }
        int begin = 0, n = 0;
        for (auto mem : hand){//判断连续的3张牌面的最大数量
            if (mem.second == 3){
                if (!begin || begin == mem.first)
                    ++n;
                if (!begin)
                    begin = mem.first;
                if (begin != mem.first && n == 1){
                    n = 1;
                    begin = mem.first;
                }
                ++begin;
            }
        }
        if (count == 3 * n){//三顺
            return TwoTriplet + ValueMax;
        }
        return Invalid;//牌不合规
    }
    case 4:
        if (count == 4){//炸弹
            return Bomb + ValueMax;
        }
        return Invalid;//牌面不合规
    default://下落，不符合规定
        return Invalid;
    }

}

int Translate(int num) {
    if (num < 52)
        return num / 4 + 3;
    else
        return num - 36;
}

template <typename Iterator>
bool next_combination(const Iterator first, Iterator k, const Iterator last)
{
   /* Credits: Mark Nelson http://marknelson.us */
   if ((first == last) || (first == k) || (last == k))
      return false;
   Iterator i1 = first;
   Iterator i2 = last;
   ++i1;
   if (last == i1)
      return false;
   i1 = last;
   --i1;
   i1 = k;
   --i2;
   while (first != i1)
   {
      if (*--i1 < *i2)
      {
         Iterator j = k;
         while (!(*i1 < *j)) ++j;
         std::iter_swap(i1,j);
         ++i1;
         ++j;
         i2 = k;
         std::rotate(i1,j,last);
         while (last != j)
         {
            ++j;
            ++i2;
         }
         std::rotate(k,i2,last);
         return true;
      }
   }
   std::rotate(first,k,last);
   return false;
}

std::vector<std::vector<int>> combinations(std::vector<int> s, int comb_size) {
    std::vector<std::vector<int>> v;
    do{
        v.push_back(std::vector<int>(s.begin(), s.begin() + comb_size));
    } while (next_combination(s.begin(), s.begin() + comb_size, s.end()));
    return v;
}

std::vector<std::vector<int>> get_action_c(std::vector<int> v) {
    std::vector<std::vector<int>> hands;
    std::map<int, int> lens;
    std::map<int, std::vector<int>> bins;
    for (auto c : v){
        ++lens[Translate(c)];
        bins[Translate(c)].push_back(c);
    }
    // single
    for (auto c : v) {
        std::vector<int> hand;
        hand.push_back(c);
        hands.push_back(hand);
    }
    // sets
    std::vector<std::vector<int>> pairs;
    std::vector<std::vector<int>> triplets;
    std::vector<std::vector<int>> bombs;
    for (auto len : lens) {
        if (len.second > 1) {
            for (auto pair : combinations(bins[len.first], 2)) {
                pairs.push_back(pair);
                hands.push_back(pair);
            }
        }
        if (len.second > 2) {
            for (auto triplet : combinations(bins[len.first], 3)) {
                triplets.push_back(triplet);
                hands.push_back(triplet);
            }
        }
        if (len.second > 3) {
            bombs.push_back(bins[len.first]);
            hands.push_back(bins[len.first]);
        }
    }
    //two triplets
    int tsize = triplets.size();
    for (int i = 0; i < tsize; ++i) {
        for (int j = i + 1; j < tsize; ++j) {
            auto triplet = triplets[i];
            auto another_triplet = triplets[j];
            if (triplet[0] / 4 == another_triplet[0] / 4 - 1 && another_triplet[0] < 48) {
                std::vector<int> two_triplet = triplet;
                two_triplet.insert(two_triplet.end(), another_triplet.begin(), another_triplet.end());
                hands.push_back(two_triplet);
            }
        }
    }
    //three pairs
    int psize = pairs.size();
    for (int i = 0; i < psize; ++i) {
        for (int j = i + 1; j < psize; ++j) {
            for (int k = j + 1; k < psize; ++k) {
                auto pair1 = pairs[i];
                auto pair2 = pairs[j];
                auto pair3 = pairs[k];
                if (pair1[0] / 4 == pair2[0] / 4 - 1 && pair2[0] / 4 == pair3[0] / 4 - 1 && pair3[0] < 48) {
                    std::vector<int> two_pair = pair1;
                    two_pair.insert(two_pair.end(), pair2.begin(), pair2.end());
                    std::vector<int> three_pair = two_pair;
                    three_pair.insert(three_pair.end(), pair3.begin(), pair3.end());
                    hands.push_back(three_pair);
                }
            }
        }
    }
    //straight
    // std::cout << "st" << std::endl;
    std::vector<std::vector<int>> straight_nums;
    int bsize = lens.size();
    std::vector<int> nums;
    for(std::map<int,int>::iterator it = lens.begin(); it != lens.end(); ++it) {
        nums.push_back(it->first);
    }
    for (int i = 0; i < bsize - 4; ++i) {
        if (nums[i + 4] == nums[i] + 4 && nums[i] + 4 < 15) {
            std::vector<int> straight_num (nums.begin() + i, nums.begin() +  i + 5);
            straight_nums.push_back(straight_num);
        }
    }
    for (auto straight_num : straight_nums) {
        std::vector<int> c1s = bins[straight_num[0]];
        std::vector<int> c2s = bins[straight_num[1]];
        std::vector<int> c3s = bins[straight_num[2]];
        std::vector<int> c4s = bins[straight_num[3]];
        std::vector<int> c5s = bins[straight_num[4]];
        for (auto c1 : c1s) {
            for (auto c2 : c2s) {
                for (auto c3 : c3s) {
                    for (auto c4 : c4s) {
                        for (auto c5 : c5s) {
                            int straight[] = {c1, c2, c3, c4, c5};
                            std::vector<int> hand(straight, straight + 5);
                            // for (auto c : hand){
                            //   std::cout << Translate(c);
                            // }
                            // std::cout << std::endl;
                            hands.push_back(hand);
                        }
                    }
                }
            }
        }
    }
    // std::cout << std::endl;
    //nuke
    if (lens.find(16) != lens.end() && lens.find(17) != lens.end()){
        std::vector<int> nuke;
        nuke.push_back(52);
        nuke.push_back(53);
        hands.push_back(nuke);
    }
    return hands;
}
